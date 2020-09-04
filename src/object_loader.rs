use std::io::Read;
use std::fs::File;
use std::io::BufReader;
//use obj::{load_obj, Obj, Position, Vertex};
//use wavefront_obj::obj::parse;
use wavefront_obj::obj::*;
//use obj::{load_obj, Obj, Position, Vertex};
use cgmath::{Vector3};
use crate::misc::*;
use crate::bvh::*;

pub fn load_triangles_from_obj(file_name: &'static str) -> Option<(Vec<Triangle>, BBox)> {

    let file_content = {
      let mut file = File::open(file_name).map_err(|e| format!("cannot open file: {}", e)).unwrap();
      let mut content = String::new();
      file.read_to_string(&mut content).unwrap();
      content
    };

    let obj_set = parse(file_content).map_err(|e| format!("cannot parse: {:?}", e)).unwrap();
    let objects = obj_set.objects;

    let mut aabb = BBox { min: Vector3::<f32>::new(0.0, 0.0, 0.0), max: Vector3::<f32>::new(0.0, 0.0, 0.0), };
    let mut result: Vec<Triangle> = Vec::new();

    if objects.len() == 1 {
        println!("objects[0].vertices.len() == {}", objects[0].vertices.len());
        for shape in &objects[0].geometry[0].shapes {
            match shape.primitive {
                Primitive::Triangle((ia, _, _), (ib, _, _), (ic, _, _)) => {

                    let vertex_a = objects[0].vertices[ia];
                    let vertex_b = objects[0].vertices[ib];
                    let vertex_c = objects[0].vertices[ic];

                    let vec_a = Vector3::<f32>::new(vertex_a.x as f32, vertex_a.y as f32, vertex_a.z as f32) * 10.0; 
                    let vec_b = Vector3::<f32>::new(vertex_b.x as f32, vertex_b.y as f32, vertex_b.z as f32) * 10.0; 
                    let vec_c = Vector3::<f32>::new(vertex_c.x as f32, vertex_c.y as f32, vertex_c.z as f32) * 10.0; 

                    aabb.expand(&vec_a);
                    aabb.expand(&vec_b);
                    aabb.expand(&vec_c);

                    let tr = Triangle {
                        a: vec_a,
                        b: vec_b,
                        c: vec_c,
                    };

                    result.push(tr);
                }
                Primitive::Line(_, _) => { panic!("load_triangles_from_obj not supporting lines."); }
                Primitive::Point(_) => { panic!("load_triangles_from_obj not supporting points."); }
            }
        }
    }

    println!("SKULL triangles.len() == {}", result.len());
    println!("SKULL objects.len() == {}", result.len());

    //let a = Vector3::<f32>::new(objects[0].vertices[ia].position[0], dome.vertices[i0].position[1], dome.vertices[i0].position[2]) * 10.0;
    //let b = Vector3::<f32>::new(objects[0].vertices[ib].position[0], dome.vertices[i1].position[1], dome.vertices[i1].position[2]) * 10.0; 
    //let c = Vector3::<f32>::new(objects[0].vertices[ia].position[0], dome.vertices[i2].position[1], dome.vertices[i2].position[2]) * 10.0;
    //aabb.expand(&a);
    //aabb.expand(&b);
    //aabb.expand(&c);
    //let tr = Triangle {
    //    a: a, 
    //    b: b, 
    //    c: c,
    //};
    //result.push(tr);

    //let input = BufReader::new(File::open(file_name).unwrap());
    ////let dome: Obj<Position> = load_obj(input).unwrap();
    //let dome: Obj<Vertex> = load_obj(input).unwrap();

    //println!("vertice.count == {}", dome.vertices.len());
    //println!("indices.count == {}", dome.indices.len());
    //for i in 0..dome.indices.len()/3 {
    //    let offset = i*3 as usize;
    //    let i0 = dome.indices[offset] as usize;  
    //    let i1 = dome.indices[offset+1] as usize;  
    //    let i2 = dome.indices[offset+2] as usize;  
    //    // println!("f {} {} {}", ia, ib, ic);
    //    let a = Vector3::<f32>::new(dome.vertices[i0].position[0], dome.vertices[i0].position[1], dome.vertices[i0].position[2]) * 10.0;
    //    let b = Vector3::<f32>::new(dome.vertices[i1].position[0], dome.vertices[i1].position[1], dome.vertices[i1].position[2]) * 10.0; 
    //    let c = Vector3::<f32>::new(dome.vertices[i2].position[0], dome.vertices[i2].position[1], dome.vertices[i2].position[2]) * 10.0;
    //    aabb.expand(&a);
    //    aabb.expand(&b);
    //    aabb.expand(&c);
    //    let tr = Triangle {
    //        a: a, 
    //        b: b, 
    //        c: c,
    //    };
    //    result.push(tr);
    //}

    Some((result, aabb))
}

//pub fn load_triangles_from_obj(file_name: &'static str) -> Option<(Vec<Triangle>, BBox)> {
//    let input = BufReader::new(File::open(file_name).unwrap());
//    //let dome: Obj<Position> = load_obj(input).unwrap();
//    let dome: Obj<Vertex> = load_obj(input).unwrap();
//
//    let mut result: Vec<Triangle> = Vec::new();
//    let mut aabb = BBox { min: Vector3::<f32>::new(0.0, 0.0, 0.0), max: Vector3::<f32>::new(0.0, 0.0, 0.0), };
//    println!("vertice.count == {}", dome.vertices.len());
//    println!("indices.count == {}", dome.indices.len());
//    for i in 0..dome.indices.len()/3 {
//        let offset = i*3 as usize;
//        let i0 = dome.indices[offset] as usize;  
//        let i1 = dome.indices[offset+1] as usize;  
//        let i2 = dome.indices[offset+2] as usize;  
//        // println!("f {} {} {}", ia, ib, ic);
//        let a = Vector3::<f32>::new(dome.vertices[i0].position[0], dome.vertices[i0].position[1], dome.vertices[i0].position[2]) * 10.0;
//        let b = Vector3::<f32>::new(dome.vertices[i1].position[0], dome.vertices[i1].position[1], dome.vertices[i1].position[2]) * 10.0; 
//        let c = Vector3::<f32>::new(dome.vertices[i2].position[0], dome.vertices[i2].position[1], dome.vertices[i2].position[2]) * 10.0;
//        aabb.expand(&a);
//        aabb.expand(&b);
//        aabb.expand(&c);
//        let tr = Triangle {
//            a: a, 
//            b: b, 
//            c: c,
//        };
//        result.push(tr);
//    }
//
//    Some((result, aabb))
//}
