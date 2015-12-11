#pragma once

#include <type_traits>
#include <iostream>
#include <cassert>

#ifndef __host__
#  define __host__
#  define VARIANT_UNDEF_HOST
#endif

#ifndef __device__
#  define __device__
#  define VARIANT_UNDEF_DEVICE
#endif


template<size_t i, typename T1, typename... Types>
  struct __initializer : __initializer<i+1,Types...>
{
  typedef __initializer<i+1,Types...> super_t;

  using super_t::operator();

  __host__ __device__
  size_t operator()(void *ptr, const T1& other)
  {
    new (ptr) T1(other);
    return i;
  }
};


template<size_t i, typename T>
struct __initializer<i,T>
{
  __host__ __device__
  size_t operator()(void* ptr, const T& other)
  {
    new (ptr) T(other);
    return i;
  }
};


template<size_t i, size_t... js>
struct __constexpr_max
{
  static const size_t value = i < __constexpr_max<js...>::value ? __constexpr_max<js...>::value : i;
};


template<size_t i>
struct __constexpr_max<i>
{
  static const size_t value = i;
};


template<typename T1, typename... Types>
class variant;


template<size_t i, typename Variant> struct variant_element;


template<size_t i, typename T0, typename... Types>
struct variant_element<i, variant<T0, Types...>>
  : variant_element<i-1,variant<Types...>>
{};


template<typename T0, typename... Types>
struct variant_element<0, variant<T0, Types...>>
{
  typedef T0 type;
};


template<size_t i, typename... Types>
using variant_element_t = typename variant_element<i,Types...>::type;


static constexpr const size_t variant_not_found = static_cast<size_t>(-1);

//template<size_t index, typename T, typename T1, typename... Types>
//struct __variant_find_impl;
//
//template<size_t index, typename T, typename T1, typename... Types>
//struct __variant_find_impl<index,T,variant<T1,Types...>>
//  : std::integral_constant<
//      size_t,
//      __variant_find_impl<index+1,T,variant<Types...>>::value
//    >
//{};
//
//template<size_t index, typename T, typename... Types>
//struct __variant_find_impl<index,T,variant<T,Types...>>
//  : std::integral_constant<
//      size_t,
//      index
//    >
//{};
//
//
//template<size_t index, typename T, typename Type>
//struct __variant_find_impl<index,T,variant<T,Type>>
//  : std::integral_constant<
//      size_t,
//      variant_not_found
//    >
//{};
//
//template<typename T, typename Variant>
//using variant_find = __variant_find_impl<0,T,Variant>;


//template<typename T, typename Variant>
//using __is_variant_alternative = std::integral_constant<bool, variant_find<T,Variant>::value != variant_not_found>;


template<typename T, typename U>
struct __propagate_reference;


template<typename T, typename U>
struct __propagate_reference<T&, U>
{
  typedef U& type;
};


template<typename T, typename U>
struct __propagate_reference<const T&, U>
{
  typedef const U& type;
};


template<typename T, typename U>
struct __propagate_reference<T&&, U>
{
  typedef U&& type;
};


template<size_t i, typename VariantReference>
struct __variant_element_reference
  : __propagate_reference<
      VariantReference,
      variant_element_t<
        i,
        typename std::decay<VariantReference>::type
      >
    >
{};


template<size_t i, typename VariantReference>
using __variant_element_reference_t = typename __variant_element_reference<i,VariantReference>::type;


template<typename Visitor, typename Variant>
__host__ __device__
auto apply_visitor(Visitor visitor, Variant&& var) ->
  typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var)>)
  >::type;


template<typename Visitor, typename Variant1, typename Variant2>
__host__ __device__
auto apply_visitor(Visitor visitor, Variant1&& var1, Variant2&& var2) ->
  typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var1)>,
            __variant_element_reference_t<0,decltype(var2)>)
  >::type;


template<size_t i, typename T, typename... Types>
struct __index_of_impl;


// no match, keep going
template<size_t i, typename T, typename U, typename... Types>
struct __index_of_impl<i,T,U,Types...>
  : __index_of_impl<i+1,T,Types...>
{};


// found a match
template<size_t i, typename T, typename... Types>
struct __index_of_impl<i,T,T,Types...>
{
  static const size_t value = i;
};


// no match
template<size_t i, typename T>
struct __index_of_impl<i,T>
{
  static const size_t value = variant_not_found;
};


template<typename T, typename... Types>
using __index_of = __index_of_impl<0,T,Types...>;


template<typename T, typename Variant>
struct __is_variant_alternative;

template<class T, class T1, class... Types>
struct __is_variant_alternative<T,variant<T1,Types...>>
  : std::integral_constant<
      bool,
      (__index_of<T,T1,Types...>::value != variant_not_found)
    >
{};


template<typename T1, typename... Types>
class variant
{
  private:
    struct destroyer
    {
      template<typename T>
      __host__ __device__
      typename std::enable_if<
        !std::is_pod<T>::value
      >::type
        operator()(T& x)
      {
        x.~T();
      }
    
      template<typename T>
      __host__ __device__
      typename std::enable_if<
        std::is_pod<T>::value
      >::type
        operator()(T& x)
      {
        // do nothing for POD types
      }
    };

  public:
    __host__ __device__
    variant() : variant(T1{}) {}

  private:
    struct placement_mover
    {
      void *ptr;

      __host__ __device__
      placement_mover(void* p) : ptr(p) {}

      template<typename U>
      __host__ __device__
      size_t operator()(U&& x)
      {
        // decay off the reference of U, if any
        using T = typename std::decay<U>::type;
        new (ptr) T(std::move(x));
        return __index_of<T,T1,Types...>::value;
      }
    };

  public:
    __host__ __device__
    variant(variant&& other)
      : index_(apply_visitor(placement_mover(data()), std::move(other)))
    {}

  private:
    struct placement_copier
    {
      void *ptr;

      __host__ __device__
      placement_copier(void* p) : ptr(p) {}

      template<typename U>
      __host__ __device__
      size_t operator()(U&& x)
      {
        // decay off the reference of U, if any
        using T = typename std::decay<U>::type;
        new (ptr) T(x);
        return __index_of<T,T1,Types...>::value;
      }
    };

  public:
    __host__ __device__
    variant(const variant& other)
      : index_(apply_visitor(placement_copier(data()), other))
    {}

    template<typename T,
             class = typename std::enable_if<
               __is_variant_alternative<T,variant>::value
             >::type>
    __host__ __device__
    variant(const T& other)
    {
      __initializer<0, T1, Types...> init;
      index_ = init(data(), other);
    }

    __host__ __device__
    ~variant()
    {
      apply_visitor(destroyer(), *this);
    }

  private:
    struct forward_assign_visitor
    {
      template<class A, class B>
      __host__ __device__
      void operator()(A& a, B&& b) const
      {
        a = std::forward<B>(b);
      }
    };

    struct destroy_and_copy_construct_visitor
    {
      template<class A, class B>
      __host__ __device__
      void operator()(A& a, const B& b) const
      {
        // copy b to a temporary
        B tmp = b;

        // destroy a
        a.~A();

        // placement move from tmp to a
        new (&a) B(std::move(tmp));
      }
    };

    struct destroy_and_move_construct_visitor
    {
      template<class A, class B>
      __host__ __device__
      void operator()(A& a, B&& b) const
      {
        // destroy a
        a.~A();

        using type = typename std::decay<B>::type;

        // placement move from b
        new (&a) type(std::move(b));
      }
    };

  public:
    __host__ __device__
    variant& operator=(const variant& other)
    {
      if(index() == other.index())
      {
        apply_visitor(forward_assign_visitor(), *this, other);
      }
      else
      {
        apply_visitor(destroy_and_copy_construct_visitor(), *this, other);
        index_ = other.index();
      }

      return *this;
    }

    __host__ __device__
    variant& operator=(variant&& other)
    {
      if(index() == other.index())
      {
        apply_visitor(forward_assign_visitor(), *this, std::move(other));
      }
      else
      {
        apply_visitor(destroy_and_move_construct_visitor(), *this, std::move(other));
        index_ = other.index();
      }

      return *this;
    }

    __host__ __device__
    size_t index() const
    {
      return index_;
    }

  private:
    struct swap_visitor
    {
      template<class A, class B>
      __host__ __device__
      void operator()(A& a, B& b)
      {
        // XXX can't call std::swap because __host__ __device__
        A tmp = a;
        a = b;
        b = tmp;
      }
    };

  public:
    __host__ __device__
    void swap(variant& other)
    {
      if(index() == other.index())
      {
        apply_visitor(swap_visitor(), *this, other);
      }
      else
      {
        variant tmp = *this;
        *this = other;
        other = std::move(tmp);
      }
    }

  private:
    struct equals
    {
      template<typename U1, typename U2>
      __host__ __device__
      bool operator()(const U1&, const U2&)
      {
        return false;
      }

      template<typename T>
      __host__ __device__
      bool operator()(const T& lhs, const T& rhs)
      {
        return lhs == rhs;
      }
    };


  public:
    __host__ __device__
    bool operator==(const variant& rhs) const
    {
      return index() == rhs.index() && apply_visitor(equals(), *this, rhs);
    }

    __host__ __device__
    bool operator!=(const variant& rhs) const
    {
      return !operator==(rhs);
    }

  private:
    struct less
    {
      template<typename U1, typename U2>
      __host__ __device__
      bool operator()(const U1&, const U2&)
      {
        return false;
      }

      template<typename T>
      __host__ __device__
      bool operator()(const T& lhs, const T& rhs)
      {
        return lhs < rhs;
      }
    };

  public:
    __host__ __device__
    bool operator<(const variant& rhs) const
    {
      if(index() != rhs.index()) return index() < rhs.index();

      return apply_visitor(less(), *this, rhs);
    }

    __host__ __device__
    bool operator<=(const variant& rhs) const
    {
      return !(rhs < *this);
    }

    __host__ __device__
    bool operator>(const variant& rhs) const
    {
      return rhs < *this;
    }

    __host__ __device__
    bool operator>=(const variant& rhs) const
    {
      return !(*this < rhs);
    }

  private:
    typedef typename std::aligned_storage<
      __constexpr_max<sizeof(T1), sizeof(Types)...>::value
    >::type storage_type;

    storage_type storage_;

    __host__ __device__
    void *data()
    {
      return &storage_;
    }

    __host__ __device__
    const void *data() const
    {
      return &storage_;
    }

    size_t index_;
};


struct __ostream_output_visitor
{
  std::ostream &os;

  __ostream_output_visitor(std::ostream& os) : os(os) {}

  template<typename T>
  std::ostream& operator()(const T& x)
  {
    return os << x;
  }
};


template<typename T1, typename... Types>
std::ostream &operator<<(std::ostream& os, const variant<T1,Types...>& v)
{
  return apply_visitor(__ostream_output_visitor(os), v);
}


template<typename Visitor, typename Result, typename T, typename... Types>
struct __apply_visitor_impl : __apply_visitor_impl<Visitor,Result,Types...>
{
  typedef __apply_visitor_impl<Visitor,Result,Types...> super_t;

  __host__ __device__
  static Result do_it(Visitor visitor, void* ptr, size_t index)
  {
    if(index == 0)
    {
      return visitor(*reinterpret_cast<T*>(ptr));
    }

    return super_t::do_it(visitor, ptr, --index);
  }


  __host__ __device__
  static Result do_it(Visitor visitor, const void* ptr, size_t index)
  {
    if(index == 0)
    {
      return visitor(*reinterpret_cast<const T*>(ptr));
    }

    return super_t::do_it(visitor, ptr, --index);
  }
};


template<typename Visitor, typename Result, typename T>
struct __apply_visitor_impl<Visitor,Result,T>
{
  __host__ __device__
  static Result do_it(Visitor visitor, void* ptr, size_t)
  {
    return visitor(*reinterpret_cast<T*>(ptr));
  }

  __host__ __device__
  static Result do_it(Visitor visitor, const void* ptr, size_t)
  {
    return visitor(*reinterpret_cast<const T*>(ptr));
  }
};


template<typename Visitor, typename Result, typename Variant>
struct __apply_visitor;


template<typename Visitor, typename Result, typename T1, typename... Types>
struct __apply_visitor<Visitor,Result,variant<T1,Types...>>
  : __apply_visitor_impl<Visitor,Result,T1,Types...>
{};


template<typename Visitor, typename Variant>
__host__ __device__
auto apply_visitor(Visitor visitor, Variant&& var) ->
  typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var)>)
  >::type
{
  using result_type = typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var)>)
  >::type;
 
  using impl = __apply_visitor<Visitor,result_type,typename std::decay<Variant>::type>;

  return impl::do_it(visitor, &var, var.index());
}



template<typename Visitor, typename Result, typename ElementReference>
struct __unary_visitor_binder
{
  Visitor visitor;
  ElementReference x;

  __host__ __device__
  __unary_visitor_binder(Visitor visitor, ElementReference x) : visitor(visitor), x(x) {}

  template<typename T>
  __host__ __device__
  Result operator()(T&& y)
  {
    return visitor(x, std::forward<T>(y));
  }
};


template<class Reference>
struct __rvalue_reference_to_lvalue_reference
{
  using type = Reference;
};

template<class T>
struct __rvalue_reference_to_lvalue_reference<T&&>
{
  using type = T&;
};


template<typename Visitor, typename Result, typename VariantReference>
struct __binary_visitor_binder
{
  Visitor visitor;
  // since rvalue references can't be members of classes, we transform any
  // VariantReference which is an rvalue reference to an lvalue reference
  // when we use y in operator(), we cast it back to the original reference type
  typename __rvalue_reference_to_lvalue_reference<VariantReference>::type y;

  __host__ __device__
  __binary_visitor_binder(Visitor visitor, VariantReference ref) : visitor(visitor), y(ref) {}

  template<typename T>
  __host__ __device__
  Result operator()(T&& x)
  {
    return apply_visitor(__unary_visitor_binder<Visitor, Result, decltype(x)>(visitor, std::forward<T>(x)), std::forward<VariantReference>(y));
  }
};


template<typename Visitor, typename Variant1, typename Variant2>
__host__ __device__
auto apply_visitor(Visitor visitor, Variant1&& var1, Variant2&& var2) ->
  typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var1)>,
            __variant_element_reference_t<0,decltype(var2)>)
  >::type
{
  using result_type = typename std::result_of<
    Visitor(__variant_element_reference_t<0,decltype(var1)>,
            __variant_element_reference_t<0,decltype(var2)>)
  >::type;

  auto visitor_wrapper = __binary_visitor_binder<Visitor,result_type,decltype(var2)>(visitor, std::forward<Variant2>(var2));

  return apply_visitor(visitor_wrapper, std::forward<Variant1>(var1));
}


template<class T>
struct __get_visitor
{
  __host__ __device__
  T* operator()(T& x) const
  {
    return &x;
  }

  __host__ __device__
  const T* operator()(const T& x) const
  {
    return &x;
  }

  template<class U>
  __host__ __device__
  T* operator()(U&&) const
  {
    return nullptr;
  }
};


template<size_t i, class T1, class... Types>
__host__ __device__
__variant_element_reference_t<i, variant<T1,Types...>&>
  get(variant<T1,Types...>& v)
{
  assert(i == v.index());

  using type = typename std::decay<
    variant_element_t<i,variant<T1,Types...>>
  >::type;

  return *apply_visitor(__get_visitor<type>(), v);
}

template<size_t i, class T1, class... Types>
__host__ __device__
__variant_element_reference_t<i, variant<T1,Types...>&&>
  get(variant<T1,Types...>&& v)
{
  assert(i == v.index());

  using type = typename std::decay<
    variant_element_t<i,variant<T1,Types...>>
  >::type;

  return std::move(*apply_visitor(__get_visitor<type>(), v));
}

template<size_t i, class T1, class... Types>
__host__ __device__
__variant_element_reference_t<i, const variant<T1,Types...>&>
  get(const variant<T1,Types...>& v)
{
  assert(i == v.index());

  using type = typename std::decay<
    variant_element_t<i,variant<T1,Types...>>
  >::type;

  return *apply_visitor(__get_visitor<type>(), v);
}


#ifdef VARIANT_UNDEF_HOST
#  undef __host__
#  undef VARIANT_UNDEF_HOST
#endif

#ifdef VARIANT_UNDEF_DEVICE
#  undef __device__
#  undef VARIANT_UNDEF_DEVICE
#endif

