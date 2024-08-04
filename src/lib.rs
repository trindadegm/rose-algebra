pub mod matrix;
pub mod vector;

pub use vector::Vector;
pub type Vector2 = Vector<2, f32>;
pub type Vector3 = Vector<3, f32>;
pub type Vector4 = Vector<4, f32>;

pub use matrix::Matrix;
pub type Matrix3 = Matrix<3, 3, f32>;
pub type Matrix4 = Matrix<4, 4, f32>;

#[rustfmt::skip]
pub mod m4 {
    use super::*;

    pub const IDENTITY: Matrix4 = Matrix4::new([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    pub const X: Matrix4 = Matrix4::new([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);
    pub const Y: Matrix4 = Matrix4::new([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);
    pub const Z: Matrix4 = Matrix4::new([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);
    pub const W: Matrix4 = Matrix4::new([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    pub const ZERO: Matrix4 = Matrix4::new([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]);

    pub fn scale(vector: &Vector4) -> Matrix4 {
        let mut matrix = ZERO;
        matrix[(0, 0)] = *vector.x();
        matrix[(1, 1)] = *vector.y();
        matrix[(2, 2)] = *vector.z();
        matrix[(3, 3)] = *vector.w();
        matrix
    }

    pub fn translate(vector: &Vector3) -> Matrix4 {
        let mut matrix = IDENTITY;
        matrix[(0, 3)] = *vector.x();
        matrix[(1, 3)] = *vector.y();
        matrix[(2, 3)] = *vector.z();
        matrix
    }

    pub fn rotate_axis(axis: &Vector3, angle: f32) -> Matrix4 {
        let sin_angle = f32::sin(angle);
        let cos_angle = f32::cos(angle);
        let axis_k_matrix = Matrix4::new([
            [0.0, -*axis.z(),*axis.y(), 0.0],
            [*axis.z(), 0.0, -*axis.x(), 0.0],
            [-*axis.y(), *axis.x(), 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);
        let rot_matrix = IDENTITY + (axis_k_matrix * sin_angle) + (axis_k_matrix.matmul(&axis_k_matrix) * (1.0 - cos_angle));
        rot_matrix
    }

    pub fn simple_projection(viewport_scale_hpv: f32, near: f32, far: f32) -> Matrix4 {
        let far_near_dif = far - near;
        Matrix4::new([
            [viewport_scale_hpv, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -far/far_near_dif, -(far * near)/far_near_dif],
            [0.0, 0.0, -1.0, 0.0],
        ])
    }

    pub fn standard_projection(viewport_scale_hpv: f32, fov_cotan: f32, near: f32, far: f32) -> Matrix4 {
        let far_near_dif = far - near;
        Matrix4::new([
            [viewport_scale_hpv * fov_cotan, 0.0, 0.0, 0.0],
            [0.0, -fov_cotan, 0.0, 0.0],
            [0.0, 0.0, -far/far_near_dif, -(far * near)/far_near_dif],
            [0.0, 0.0, -1.0, 0.0],
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_add_sub_test() {
        let a = Vector3::new([0.0, 1.0, 2.0]);
        let b = Vector3::new([-2.0, 3.0, -4.0]);
        let c = Vector3::new([0.0 - 2.0, 1.0 + 3.0, 2.0 - 4.0]);

        assert_eq!(a + b, c);
        assert_eq!(c - b, a);
        assert_eq!(c - a, b);

        let mut a2 = a;
        let mut c2 = c;
        let mut c3 = c;

        a2 += b;
        c2 -= b;
        c3 -= a;

        assert_eq!(a2, c);
        assert_eq!(c2, a);
        assert_eq!(c3, b);
    }

    #[test]
    fn vector_length_test() {
        let v2_a = Vector2::new([1.0, 1.0]);
        let v2_b = Vector2::new([1.0, -1.0]);

        assert_eq!(v2_a.length(), f32::sqrt(2.0));
        assert_eq!(v2_b.length(), f32::sqrt(2.0));
        assert_eq!(v2_a.length_squared(), 2.0);
        assert_eq!(v2_b.length_squared(), 2.0);

        let v3_a = Vector3::new([1.0, 2.0, 3.5]);
        let v4_a = Vector4::new([3.0, 4.0, 5.0, -6.0]);

        assert_eq!(v3_a.length(), f32::sqrt(1.0 + 4.0 + (3.5 * 3.5)));
        assert_eq!(v3_a.length_squared(), 1.0 + 4.0 + (3.5 * 3.5));
        assert_eq!(
            v4_a.length(),
            f32::sqrt((3.0 * 3.0) + (4.0 * 4.0) + (5.0 * 5.0) + (-6.0 * -6.0))
        );
        assert_eq!(
            v4_a.length_squared(),
            (3.0 * 3.0) + (4.0 * 4.0) + (5.0 * 5.0) + (-6.0 * -6.0)
        );
    }

    #[test]
    fn vector_scaling_test() {
        let v2_a = Vector2::new([1.0, 1.0]);
        let v3_a = Vector3::new([1.0, 2.0, 3.5]);
        let v4_a = Vector4::new([3.0, 4.0, 5.0, -6.0]);

        assert_eq!(v2_a * 2.0, Vector2::new([2.0, 2.0]));
        assert_eq!(v2_a / 3.0, Vector2::new([1.0 / 3.0, 1.0 / 3.0]));

        assert_eq!(v3_a * 2.0, Vector3::new([2.0, 4.0, 3.5 * 2.0]));
        assert_eq!(v3_a / 3.0, Vector3::new([1.0 / 3.0, 2.0 / 3.0, 3.5 / 3.0]));

        assert_eq!(
            v4_a * 2.0,
            Vector4::new([3.0 * 2.0, 4.0 * 2.0, 5.0 * 2.0, -6.0 * 2.0])
        );
        assert_eq!(
            v4_a / 3.0,
            Vector4::new([3.0 / 3.0, 4.0 / 3.0, 5.0 / 3.0, -6.0 / 3.0])
        );

        let mut v2_a_2 = v2_a;
        let mut v2_a_3 = v2_a;

        v2_a_2 *= 2.0;
        v2_a_3 /= 3.0;

        assert_eq!(v2_a_2, Vector2::new([2.0, 2.0]));
        assert_eq!(v2_a_3, Vector2::new([1.0 / 3.0, 1.0 / 3.0]));

        let mut v3_a_2 = v3_a;
        let mut v3_a_3 = v3_a;

        v3_a_2 *= 2.0;
        v3_a_3 /= 3.0;

        assert_eq!(v3_a_2, Vector3::new([2.0, 4.0, 3.5 * 2.0]));
        assert_eq!(v3_a_3, Vector3::new([1.0 / 3.0, 2.0 / 3.0, 3.5 / 3.0]));

        let mut v4_a_2 = v4_a;
        let mut v4_a_3 = v4_a;

        v4_a_2 *= 2.0;
        v4_a_3 /= 3.0;

        assert_eq!(
            v4_a_2,
            Vector4::new([3.0 * 2.0, 4.0 * 2.0, 5.0 * 2.0, -6.0 * 2.0])
        );
        assert_eq!(
            v4_a_3,
            Vector4::new([3.0 / 3.0, 4.0 / 3.0, 5.0 / 3.0, -6.0 / 3.0])
        );
    }

    #[test]
    fn vector_product_test() {
        let v2_a = Vector2::new([4.1, 4.2]);
        let v2_b = Vector2::new([5.3, 5.4]);
        let v2_dot = v2_a.x() * v2_b.x() + v2_a.y() * v2_b.y();

        assert_eq!(v2_a.dot(v2_b), v2_dot);
        assert_eq!(v2_b.dot(v2_a), v2_dot);

        let v3_a = Vector3::new([1.0, 2.0, 3.5]);
        let v3_b = Vector3::new([-2.1, 2.3, -3.4]);
        let v3_dot = v3_a.x() * v3_b.x() + v3_a.y() * v3_b.y() + v3_a.z() * v3_b.z();
        let v3_ab_cross = Vector3::new([-14.85, -3.95, 6.5]);
        let v3_ba_cross = Vector3::new([14.85, 3.95, -6.5]);

        assert_eq!(v3_a.dot(v3_b), v3_dot);
        assert_eq!(v3_b.dot(v3_a), v3_dot);
        assert!((v3_a.cross(v3_b) - v3_ab_cross)
            .data()
            .into_iter()
            .all(|x| x.abs() < 0.0001));
        assert!((v3_b.cross(v3_a) - v3_ba_cross)
            .data()
            .into_iter()
            .all(|x| x.abs() < 0.0001));

        let v4_a = Vector4::new([3.0, 4.0, 5.0, -6.0]);
        let v4_b = Vector4::new([2.0, 3.0, -4.0, -5.0]);
        let v4_dot =
            v4_a.x() * v4_b.x() + v4_a.y() * v4_b.y() + v4_a.z() * v4_b.z() + v4_a.w() * v4_b.w();

        assert_eq!(v4_a.dot(v4_b), v4_dot);
        assert_eq!(v4_b.dot(v4_a), v4_dot);
    }

    #[test]
    fn vector_normalized_test() {
        let v2_a = Vector2::new([1.0, 1.0]);

        let v3_a = Vector3::new([1.0, 2.0, 3.5]);
        let v3_b = Vector3::new([0.0, 0.0, 0.0]);

        let v4_a = Vector4::new([3.0, 4.0, 5.0, -6.0]);

        assert_eq!(v2_a.normalized(), v2_a / v2_a.length());
        assert_eq!(v3_a.normalized(), v3_a / v3_a.length());
        assert_eq!(v3_b.normalized(), Vector::new([0.0, 0.0, 0.0]));
        assert_eq!(v4_a.normalized(), v4_a / v4_a.length());
    }

    #[test]
    fn matrix_add_sub_test() {
        let a = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let b = Matrix::new([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]);
        let c = Matrix::new([[2.5, 4.5, 6.5], [8.5, 10.5, 12.5]]);

        assert_eq!(a + b, c);
        assert_eq!(c - b, a);
        assert_eq!(c - a, b);

        let mut a2 = a;
        let mut c2 = c;
        let mut c3 = c;

        a2 += b;
        c2 -= b;
        c3 -= a;

        assert_eq!(a2, c);
        assert_eq!(c2, a);
        assert_eq!(c3, b);
    }

    #[test]
    fn matrix_scaling_test() {
        let v2_a = Matrix::new([[1.0, 1.0]]);
        let v3_a = Matrix::new([[1.0, 2.0, 3.5]]);
        let v4_a = Matrix::new([[3.0, 4.0], [5.0, -6.0]]);

        assert_eq!(v2_a * 2.0, Matrix::new([[2.0, 2.0]]));
        assert_eq!(v2_a / 3.0, Matrix::new([[1.0 / 3.0, 1.0 / 3.0]]));

        assert_eq!(v3_a * 2.0, Matrix::new([[2.0, 4.0, 3.5 * 2.0]]));
        assert_eq!(v3_a / 3.0, Matrix::new([[1.0 / 3.0, 2.0 / 3.0, 3.5 / 3.0]]));

        assert_eq!(
            v4_a * 2.0,
            Matrix::new([[3.0 * 2.0, 4.0 * 2.0], [5.0 * 2.0, -6.0 * 2.0]])
        );
        assert_eq!(
            v4_a / 3.0,
            Matrix::new([[3.0 / 3.0, 4.0 / 3.0], [5.0 / 3.0, -6.0 / 3.0]])
        );

        let mut v2_a_2 = v2_a;
        let mut v2_a_3 = v2_a;

        v2_a_2 *= 2.0;
        v2_a_3 /= 3.0;

        assert_eq!(v2_a_2, Matrix::new([[2.0, 2.0]]));
        assert_eq!(v2_a_3, Matrix::new([[1.0 / 3.0, 1.0 / 3.0]]));

        let mut v3_a_2 = v3_a;
        let mut v3_a_3 = v3_a;

        v3_a_2 *= 2.0;
        v3_a_3 /= 3.0;

        assert_eq!(v3_a_2, Matrix::new([[2.0, 4.0, 3.5 * 2.0]]));
        assert_eq!(v3_a_3, Matrix::new([[1.0 / 3.0, 2.0 / 3.0, 3.5 / 3.0]]));

        let mut v4_a_2 = v4_a;
        let mut v4_a_3 = v4_a;

        v4_a_2 *= 2.0;
        v4_a_3 /= 3.0;

        assert_eq!(
            v4_a_2,
            Matrix::new([[3.0 * 2.0, 4.0 * 2.0], [5.0 * 2.0, -6.0 * 2.0]])
        );
        assert_eq!(
            v4_a_3,
            Matrix::new([[3.0 / 3.0, 4.0 / 3.0], [5.0 / 3.0, -6.0 / 3.0]])
        );
    }

    #[test]
    fn matrix_multiply_test() {
        let mat_a = Matrix::new([[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]]);
        let mat_b = Matrix::new([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]);

        let res = mat_a.matmul(&mat_b);

        assert_eq!(
            res,
            Matrix::new([
                [
                    (1.0 * 0.1 + 2.0 * 0.3 + 3.0 * 0.5),
                    (1.0 * 0.2 + 2.0 * 0.4 + 3.0 * 0.6)
                ],
                [
                    (5.0 * 0.1 + 6.0 * 0.3 + 7.0 * 0.5),
                    (5.0 * 0.2 + 6.0 * 0.4 + 7.0 * 0.6)
                ],
            ]),
        );

        let mat_a = Matrix::new([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let mat_b = Matrix::new([[1.0], [2.0], [3.0], [4.0]]);

        let res = mat_a.matmul(&mat_b);
        assert_eq!(
            res,
            Matrix::new([
                [1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0],
                [5.0 * 1.0 + 6.0 * 2.0 + 7.0 * 3.0 + 8.0 * 4.0],
                [9.0 * 1.0 + 10.0 * 2.0 + 11.0 * 3.0 + 12.0 * 4.0],
                [13.0 * 1.0 + 14.0 * 2.0 + 15.0 * 3.0 + 16.0 * 4.0],
            ]),
        );
    }
}
