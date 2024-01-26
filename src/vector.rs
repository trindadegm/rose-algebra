use std::{
    borrow::Borrow,
    iter::Sum,
    mem::MaybeUninit,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign, Index, IndexMut},
};

use crate::Matrix;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Vector<const N: usize, T>([T; N]);

impl<const N: usize, T> Vector<N, T> {
    #[inline]
    pub const fn new(values: [T; N]) -> Self {
        Self(values)
    }

    /// Returns a reference to the array of vector elements.
    #[inline(always)]
    pub fn data(&self) -> &[T; N] {
        &self.0
    }

    #[inline(always)]
    pub fn take_data(self) -> [T; N] {
        self.0
    }
}
impl<const N: usize, T> Index<usize> for Vector<N, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}
impl<const N: usize, T> IndexMut<usize> for Vector<N, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
impl<const N: usize, T> Default for Vector<N, T>
where
    T: Default,
{
    fn default() -> Self {
        if N > 0 {
            let mut data = MaybeUninit::<[T; N]>::uninit();
            // This is safe because we have exclusive access to `data`'s memory,
            // which is wrapped on a `MaybeUninit`, so that the compiler does
            // not care about it, but the memory is still there.
            //
            // The alignment is correct because `MaybeUninit` will take care of
            // alignment requirements. The pointer provenance is fine, because
            // all of the memory accessed belongs to the same `MaybeUninit`.
            //
            // We take a pointer `*mut [T; N]` and convert to a `*mut T`. As we
            // make sure N is not 0, any pointer to the array is a valid pointer
            // to T, and we can offset it based on N, which we do.
            //
            // As the memory is uninitialized, we cannot read from it, so we use
            // the `write` method only.
            let data = unsafe {
                let elem_ptr = data.as_mut_ptr() as *mut T;
                for i in 0..(TryInto::<isize>::try_into(N).unwrap()) {
                    elem_ptr.offset(i).write(T::default());
                }
                data.assume_init()
            };
            Self(data)
        } else {
            // This does not allocate so the compiler shoudn't do anything and
            // optimize it away (hopefully)
            Self(Vec::new().try_into().map_err(|_| ()).unwrap())
        }
    }
}

macro_rules! impl_op {
    ($trait_name:ident, $opname:ident) => {
        impl<const N: usize, T, V> $trait_name<V> for Vector<N, T>
        where
            T: Default,
            for<'a> &'a T: $trait_name<&'a T, Output = T>,
            V: Borrow<Vector<N, T>>,
        {
            type Output = Vector<N, T>;

            #[inline(always)]
            fn $opname(self, rhs: V) -> Self::Output {
                (&self).$opname(rhs)
            }
        }
        impl<const N: usize, T, V> $trait_name<V> for &Vector<N, T>
        where
            T: Default,
            for<'a> &'a T: $trait_name<&'a T, Output = T>,
            V: Borrow<Vector<N, T>>,
        {
            type Output = Vector<N, T>;

            fn $opname(self, rhs: V) -> Self::Output {
                let mut result: Vector<N, T> = Default::default();
                result
                    .0
                    .iter_mut()
                    .zip(self.0.iter().zip(rhs.borrow().0.iter()))
                    .for_each(|(v, (a, b))| *v = a.$opname(b));
                result
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);

// With Assign

// AddAssign

impl<const N: usize, T, V> AddAssign<V> for Vector<N, T>
where
    for<'a> T: AddAssign<&'a T>,
    V: Borrow<Vector<N, T>>,
{
    fn add_assign(&mut self, rhs: V) {
        self.0
            .iter_mut()
            .zip(rhs.borrow().0.iter())
            .for_each(|(left, right)| *left += right);
    }
}

// SubAssign

impl<const N: usize, T, V> SubAssign<V> for Vector<N, T>
where
    for<'a> T: SubAssign<&'a T>,
    V: Borrow<Vector<N, T>>,
{
    fn sub_assign(&mut self, rhs: V) {
        self.0
            .iter_mut()
            .zip(rhs.borrow().0.iter())
            .for_each(|(left, right)| *left -= right);
    }
}

// Multiply by scalar

macro_rules! impl_scalar_op {
    ($trait_name:ident, $opname:ident) => {
        impl<const N: usize, T, S> $trait_name<S> for Vector<N, T>
        where
            T: Default,
            for<'a> &'a T: $trait_name<&'a T, Output = T>,
            S: Borrow<T>,
        {
            type Output = Vector<N, T>;

            #[inline(always)]
            fn $opname(self, rhs: S) -> Self::Output {
                (&self).$opname(rhs)
            }
        }
        impl<const N: usize, T, S> $trait_name<S> for &Vector<N, T>
        where
            T: Default,
            for<'a> &'a T: $trait_name<&'a T, Output = T>,
            S: Borrow<T>,
        {
            type Output = Vector<N, T>;

            fn $opname(self, rhs: S) -> Self::Output {
                let mut result = Self::Output::default();
                result
                    .0
                    .iter_mut()
                    .zip(self.0.iter())
                    .for_each(|(res, lhs)| *res = lhs.$opname(rhs.borrow()));
                result
            }
        }
    };
}

impl_scalar_op!(Mul, mul);
impl_scalar_op!(Div, div);

// With Assign

// MulAssign

impl<const N: usize, T, S> MulAssign<S> for Vector<N, T>
where
    for<'a> T: MulAssign<&'a T>,
    S: Borrow<T>,
{
    fn mul_assign(&mut self, rhs: S) {
        self.0.iter_mut().for_each(|el| *el *= rhs.borrow());
    }
}

// DivAssign

impl<const N: usize, T, S> DivAssign<S> for Vector<N, T>
where
    for<'a> T: DivAssign<&'a T>,
    S: Borrow<T>,
{
    fn div_assign(&mut self, rhs: S) {
        self.0.iter_mut().for_each(|el| *el /= rhs.borrow());
    }
}

// Common vector stuff

impl<const N: usize, T> Vector<N, T>
where
    T: Sum,
    for<'a> &'a T: Mul<&'a T, Output = T>,
{
    /// Returns the vector magnitude squared.
    ///
    /// This function avoids a square-root operation that would occur when
    /// calling [length][`Self::length()`], meaning it should perform better,
    /// although it is obviously not a replacement.
    ///
    /// It is recommended the programmer call this function instead of doing
    /// something like `v.length() * v.length()`, or `let vl = v.length(); vl *
    /// vl`.
    pub fn length_squared(&self) -> T {
        self.0.iter().map(|x| x * x).sum()
    }

    /// Calculates the scalar product of the two vectors.
    pub fn dot(&self, other: impl AsRef<Self>) -> T {
        self.0
            .iter()
            .zip(other.as_ref().0.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}
impl<const N: usize, T> Vector<N, T>
where
    T: Sqrt<Output = T> + Sum + PartialEq + Default,
    for<'a> &'a T: Mul<&'a T, Output = T> + Div<&'a T, Output = T>,
{
    /// Returns the vector magnitude.
    pub fn length(&self) -> T {
        self.0.iter().map(|x| x * x).sum::<T>().sqrt()
    }

    /// When the length of `self` is not 0, returns a vector parallel to `self`
    /// but of length 1. When the length of `self` is 0, returns a zero vector.
    pub fn normalized(&self) -> Self {
        let length = self.length();
        if length != Default::default() {
            self / length
        } else {
            Self::default()
        }
    }
}
impl<T> Vector<3, T>
where
    for<'a> &'a T: Mul<&'a T, Output = T> + Sub<&'a T, Output = T>,
{
    /// Calculates the cross product of the two vectors.
    pub fn cross(&self, other: impl AsRef<Self>) -> Self {
        let other = other.as_ref();
        Self([
            &(&self.0[1] * &other.0[2]) - &(&self.0[2] * &other.0[1]),
            &(&self.0[2] * &other.0[0]) - &(&self.0[0] * &other.0[2]),
            &(&self.0[0] * &other.0[1]) - &(&self.0[1] * &other.0[0]),
        ])
    }
}
impl<T> Vector<2, T> {
    #[inline(always)]
    pub fn x(&self) -> &T {
        &self.0[0]
    }

    #[inline(always)]
    pub fn y(&self) -> &T {
        &self.0[1]
    }
}
impl<T> Vector<3, T> {
    #[inline(always)]
    pub fn x(&self) -> &T {
        &self.0[0]
    }

    #[inline(always)]
    pub fn y(&self) -> &T {
        &self.0[1]
    }

    #[inline(always)]
    pub fn z(&self) -> &T {
        &self.0[2]
    }
}
impl<T> Vector<4, T> {
    #[inline(always)]
    pub fn x(&self) -> &T {
        &self.0[0]
    }

    #[inline(always)]
    pub fn y(&self) -> &T {
        &self.0[1]
    }

    #[inline(always)]
    pub fn z(&self) -> &T {
        &self.0[2]
    }

    #[inline(always)]
    pub fn w(&self) -> &T {
        &self.0[3]
    }
}
impl<const N: usize, T> AsRef<Vector<N, T>> for Vector<N, T> {
    #[inline(always)]
    fn as_ref(&self) -> &Self {
        self
    }
}
impl<const N: usize, T> AsRef<[T; N]> for Vector<N, T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T; N] {
        self.data()
    }
}
impl<const N: usize, T> From<Matrix<N, 1, T>> for Vector<N, T>
where T: Default + Copy,
{
    fn from(value: Matrix<N, 1, T>) -> Self {
        let mut data = [T::default(); N];
        for (i, value) in value.data().iter().enumerate() {
            data[i] = value[0];
        }
        Self::new(data)
    }
}

// Implement stuff for some std types

pub trait Sqrt {
    type Output;

    fn sqrt(self) -> Self::Output;
}

impl Sqrt for f32 {
    type Output = Self;

    #[inline(always)]
    fn sqrt(self) -> Self::Output {
        self.sqrt()
    }
}
impl Sqrt for &f32 {
    type Output = f32;

    #[inline(always)]
    fn sqrt(self) -> Self::Output {
        f32::sqrt(*self)
    }
}
impl Sqrt for f64 {
    type Output = Self;

    #[inline(always)]
    fn sqrt(self) -> Self::Output {
        self.sqrt()
    }
}
impl Sqrt for &f64 {
    type Output = f64;

    #[inline(always)]
    fn sqrt(self) -> Self::Output {
        f64::sqrt(*self)
    }
}
