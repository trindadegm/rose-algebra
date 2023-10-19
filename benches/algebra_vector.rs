use std::ops::Add;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rose_algebra::vector::Vector;

#[derive(Clone, Copy, PartialEq, Debug)]
#[repr(transparent)]
pub struct CommonVec3f32(pub [f32; 3]);

impl Add for CommonVec3f32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        CommonVec3f32([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}

pub type TemplatedVec3f32 = Vector<3, f32>;

pub const VEC3_F32_DATA_1: [f32; 3] = [4.1, -2.3, -14.0123];
pub const VEC3_F32_DATA_2: [f32; 3] = [123.0, 456.0, 789.0];

fn vec_add_benchmark(c: &mut Criterion) {
    let common_vec3_1 = CommonVec3f32(VEC3_F32_DATA_1);
    let common_vec3_2 = CommonVec3f32(VEC3_F32_DATA_2);

    let mut group = c.benchmark_group("vec3_f32_add");

    group.bench_function("common_vec3f32_add", |b| {
        b.iter(|| black_box(common_vec3_1) + black_box(common_vec3_2))
    });

    let templated_vec3_1 = TemplatedVec3f32::new(VEC3_F32_DATA_1);
    let templated_vec3_2 = TemplatedVec3f32::new(VEC3_F32_DATA_2);

    group.bench_function("templated_vec3f32_cc_add", |b| {
        b.iter(|| black_box(templated_vec3_1.clone()) + black_box(templated_vec3_2.clone()))
    });

    group.bench_function("templated_vec3f32_bc_add", |b| {
        b.iter(|| black_box(&templated_vec3_1) + black_box(templated_vec3_2.clone()))
    });

    group.bench_function("templated_vec3f32_bb_add", |b| {
        b.iter(|| black_box(&templated_vec3_1) + black_box(&templated_vec3_2))
    });

    group.bench_function("templated_vec3f32_cb_add", |b| {
        b.iter(|| black_box(templated_vec3_1.clone()) + black_box(&templated_vec3_2))
    });
}

criterion_group!(benches, vec_add_benchmark);
criterion_main!(benches);
