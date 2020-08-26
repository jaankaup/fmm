use std::fs::File;
use std::io::BufReader;
use obj::{load_obj, Obj, Position};
use cgmath::{Vector3};
use crate::misc::*;
use crate::bvh::*;

pub fn load_triangles_from_obj(file_name: &'static str) -> Option<(Vec<Triangle>, BBox)> {
    let input = BufReader::new(File::open(file_name).unwrap());
    let dome: Obj<Position> = load_obj(input).unwrap();

    let mut result: Vec<Triangle> = Vec::new();
    let mut aabb = BBox { min: Vector3::<f32>::new(0.0, 0.0, 0.0), max: Vector3::<f32>::new(0.0, 0.0, 0.0), };
    println!("vertice.count == {}", dome.vertices.len());
    for i in 0..dome.indices.len()/3 {
        let offset = i*3 as usize;
        let i0 = dome.indices[offset] as usize;  
        let i1 = dome.indices[offset+1] as usize;  
        let i2 = dome.indices[offset+2] as usize;  
        // println!("f {} {} {}", ia, ib, ic);
        let a = Vector3::<f32>::new(dome.vertices[i0].position[0], dome.vertices[i0].position[1], dome.vertices[i0].position[2]) * 10.0;
        let b = Vector3::<f32>::new(dome.vertices[i1].position[0], dome.vertices[i1].position[1], dome.vertices[i1].position[2]) * 10.0; 
        let c = Vector3::<f32>::new(dome.vertices[i2].position[0], dome.vertices[i2].position[1], dome.vertices[i2].position[2]) * 10.0;
        aabb.expand(&a);
        aabb.expand(&b);
        aabb.expand(&c);
        let tr = Triangle {
            a: a, 
            b: b, 
            c: c,
        };
        result.push(tr);
    }
    Some((result, aabb))
}
