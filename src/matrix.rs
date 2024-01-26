use std::borrow::Borrow;
use std::mem::MaybeUninit;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::Vector;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Matrix<const N: usize, const M: usize, T>([[T; M]; N]);

impl<const N: usize, const M: usize, T> Matrix<N, M, T> {
    #[inline]
    pub const fn new(values: [[T; M]; N]) -> Self {
        Matrix(values)
    }

    /// Returns a reference to the array of matrix elements.
    #[inline(always)]
    pub const fn data(&self) -> &[[T; M]; N] {
        &self.0
    }

    /// Returns a slice of the matrix elements (row-major)
    #[inline(always)]
    pub fn elements(&self) -> &[T] {
        // This should be fine. As self.data() returns a reference, we can
        // assume the pointer is non-null and aligned even it the size is 0.
        // The lifetime is also fine.
        unsafe { std::slice::from_raw_parts(self.data() as *const [T; M] as *const T, N * M) }
    }

    #[inline(always)]
    pub const fn num_rows(&self) -> usize {
        N
    }

    #[inline(always)]
    pub const fn num_cols(&self) -> usize {
        M
    }
}
impl<const N: usize, const M: usize, T> Default for Matrix<N, M, T>
where
    T: Default,
{
    fn default() -> Self {
        if N > 0 && M > 0 {
            let mut data = MaybeUninit::<[[T; M]; N]>::uninit();
            // This is safe because we have exclusive access to `data`'s memory,
            // which is wrapped on a `MaybeUninit`, so that the compiler does
            // not care about it, but the memory is still there.
            //
            // The alignment is correct because `MaybeUninit` will take care of
            // alignment requirements. The pointer provenance is fine, because
            // all of the memory accessed belongs to the same `MaybeUninit`.
            //
            // We take a pointer `*mut [[T; M]; N]` and convert to a `*mut T`.
            // As we make sure N and M are not 0, any pointer to the array is a
            // valid pointer to T, and we can offset it based on N and M, which
            // we do.
            //
            // As the memory is uninitialized, we cannot read from it, so we use
            // the `write` method only.
            let data = unsafe {
                let elem_ptr = data.as_mut_ptr() as *mut T;
                let nrows = TryInto::<isize>::try_into(N).unwrap();
                let ncols = TryInto::<isize>::try_into(M).unwrap();
                for i in 0..nrows {
                    for j in 0..ncols {
                        elem_ptr.offset(i * ncols + j).write(T::default());
                    }
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
impl<const N: usize, const M: usize, T> Index<(usize, usize)> for Matrix<N, M, T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.0[index.0][index.1]
    }
}
impl<const N: usize, const M: usize, T> IndexMut<(usize, usize)> for Matrix<N, M, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.0[index.0][index.1]
    }
}

macro_rules! impl_op {
    ($trait_name:ident, $opname:ident) => {
        impl<const N: usize, const M: usize, T, V> $trait_name<V> for Matrix<N, M, T>
        where
            T: Default,
            for<'a> &'a T: $trait_name<&'a T, Output = T>,
            V: Borrow<Matrix<N, M, T>>,
        {
            type Output = Matrix<N, M, T>;

            #[inline(always)]
            fn $opname(self, rhs: V) -> Self::Output {
                (&self).$opname(rhs)
            }
        }
        impl<const N: usize, const M: usize, T, V> $trait_name<V> for &Matrix<N, M, T>
        where
            T: Default,
            for<'a> &'a T: $trait_name<&'a T, Output = T>,
            V: Borrow<Matrix<N, M, T>>,
        {
            type Output = Matrix<N, M, T>;

            fn $opname(self, rhs: V) -> Self::Output {
                let mut result: Matrix<N, M, T> = Default::default();
                result
                    .0
                    .iter_mut()
                    .flatten()
                    .zip(self.0.iter().flatten().zip(rhs.borrow().0.iter().flatten()))
                    .for_each(|(v, (a, b))| *v = a.$opname(b));
                result
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);

impl<const N: usize, const M: usize, T, V> AddAssign<V> for Matrix<N, M, T>
where
    T: Default,
    for<'a> T: AddAssign<&'a T>,
    V: Borrow<Matrix<N, M, T>>,
{
    fn add_assign(&mut self, rhs: V) {
        self.0
            .iter_mut()
            .flatten()
            .zip(rhs.borrow().0.iter().flatten())
            .for_each(|(left, right)| {
                *left += right;
            });
    }
}
impl<const N: usize, const M: usize, T, V> SubAssign<V> for Matrix<N, M, T>
where
    T: Default,
    for<'a> T: SubAssign<&'a T>,
    V: Borrow<Matrix<N, M, T>>,
{
    fn sub_assign(&mut self, rhs: V) {
        self.0
            .iter_mut()
            .flatten()
            .zip(rhs.borrow().0.iter().flatten())
            .for_each(|(left, right)| {
                *left -= right;
            });
    }
}

// Scalar multiplication

macro_rules! impl_scalar_op {
    ($trait_name:ident, $opname:ident) => {
        impl<const N: usize, const M: usize, T, S> $trait_name<S> for Matrix<N, M, T>
        where
            T: Default,
            for<'a> &'a T: $trait_name<&'a T, Output = T>,
            S: Borrow<T>,
        {
            type Output = Matrix<N, M, T>;

            #[inline(always)]
            fn $opname(self, rhs: S) -> Self::Output {
                (&self).$opname(rhs)
            }
        }
        impl<const N: usize, const M: usize, T, S> $trait_name<S> for &Matrix<N, M, T>
        where
            T: Default,
            for<'a> &'a T: $trait_name<&'a T, Output = T>,
            S: Borrow<T>,
        {
            type Output = Matrix<N, M, T>;

            fn $opname(self, rhs: S) -> Self::Output {
                let mut result: Self::Output = Default::default();
                result
                    .0
                    .iter_mut()
                    .flatten()
                    .zip(self.0.iter().flatten())
                    .for_each(|(result_v, self_v)| *result_v = self_v.$opname(rhs.borrow()));
                result
            }
        }
    };
}

impl_scalar_op!(Mul, mul);
impl_scalar_op!(Div, div);

// Scalar multiplication with assign

impl<const N: usize, const M: usize, T, S> MulAssign<S> for Matrix<N, M, T>
where
    T: Default,
    for<'a> T: MulAssign<&'a T>,
    S: Borrow<T>,
{
    fn mul_assign(&mut self, rhs: S) {
        self.0
            .iter_mut()
            .flatten()
            .for_each(|res| *res *= rhs.borrow());
    }
}
impl<const N: usize, const M: usize, T, S> DivAssign<S> for Matrix<N, M, T>
where
    T: Default,
    for<'a> T: DivAssign<&'a T>,
    S: Borrow<T>,
{
    fn div_assign(&mut self, rhs: S) {
        self.0
            .iter_mut()
            .flatten()
            .for_each(|res| *res /= rhs.borrow());
    }
}

// Matrix multiplication

impl<const N: usize, const M: usize, T> Matrix<N, M, T>
where
    T: Default,
    for<'a> &'a T: Mul<&'a T, Output = T>,
    for<'a> T: AddAssign<&'a T>,
{
    pub fn matmul<const N2: usize>(&self, rhs: &Matrix<M, N2, T>) -> Matrix<N, N2, T> {
        let mut result: Matrix<N, N2, T> = Default::default();

        for i in 0..N {
            for j in 0..N2 {
                let mut sum = T::default();
                for k in 0..M {
                    sum += &(&self[(i, k)] * &rhs[(k, j)]);
                }
                result[(i, j)] = sum;
            }
        }

        result
    }
}

impl<const N: usize, T> From<Vector<N, T>> for Matrix<N, 1, T> {
    #[inline]
    fn from(vector: Vector<N, T>) -> Self {
        if N > 0 {
            let mut res_array = MaybeUninit::<[[T; 1]; N]>::uninit();

            // This is safe because we have exclusive access to `res_arrays`'s
            // memory, which is wrapped on a `MaybeUninit`, so that the compiler
            // does not care about it, but the memory is still there.
            //
            // The alignment is correct because `MaybeUninit` will take care of
            // alignment requirements. The pointer provenance is fine, because all
            // of the memory accessed belongs to the same `MaybeUninit`.
            //
            // We take a pointer `*mut [[T; M]; N]` and convert to a `*mut T`. As we
            // make sure N and M are not 0, any pointer to the array is a valid
            // pointer to T, and we can offset it based on N and M, which we do.
            //
            // As the memory is uninitialized, we cannot read from it, so we use the
            // `write` method only.
            let res_array = unsafe {
                let res_ptr = res_array.as_mut_ptr() as *mut T;

                for (i, v) in vector.take_data().into_iter().enumerate() {
                    res_ptr.offset(i.try_into().unwrap()).write(v);
                }

                res_array.assume_init()
            };

            Matrix::new(res_array)
        } else {
            Matrix::new(Vec::new().try_into().map_err(|_| ()).unwrap())
        }
    }
}
