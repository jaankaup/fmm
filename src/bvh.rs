use cgmath::{prelude::*, Vector3};
//use cgmath::structure::InnerSpace;

/// A struct for aabb.
pub struct BBox {
    min: Vector3<f32>,
    max: Vector3<f32>,
}

pub struct Triangle {
    pos: Vector3<f32>,
}

pub struct Plane {
    n: Vector3<f32>,    
    d: f32,    
}

impl Plane {
    pub fn new(a: &Vector3<f32>, b: &Vector3<f32>, c: &Vector3<f32>) -> Self {
        let n = (b-a).cross(c-a).normalize();
        let d = n.dot(*a);
        Self {
            n: n,
            d: d,
        }
    }

    pub fn closest_point_to_plane(self, q: &Vector3<f32>) -> Vector3<f32> {
        let t = (self.n.dot(*q) - self.d) / self.n.dot(self.n);
        q - t * self.n
    }
}

impl BBox {
    
    /// Create BBox from three vectors.
    pub fn new(a: &Vector3<f32>, b: &Vector3<f32>, c: &Vector3<f32>) -> Self {

        let mut min = min_vec(a, b);
        min = min_vec(&min, c);

        let mut max = max_vec(a, b);
        max = max_vec(&max, c);

        Self {
            min: min,
            max: max,
        }
    }

    /// Expand BBox to include point p.
    pub fn expand(&mut self, p: &Vector3<f32>) {
        self.min = Vector3::<f32>::new(self.min.x.min(p.x), self.min.y.min(p.y), self.min.z.min(p.z));
        self.max = Vector3::<f32>::new(self.max.x.max(p.x), self.max.y.max(p.y), self.max.z.max(p.z));
    }

    pub fn to_lines(&self) -> Vec<f32> {

        let dx = self.max.x - self.min.x;
        let dy = self.max.y - self.min.y;
        let dz = self.max.z - self.min.z;

        let p0 = self.min + Vector3::<f32>::new(0.0           , 0.0           , dz);
        let p1 = self.min + Vector3::<f32>::new(0.0           , dy            , dz);
        let p2 = self.min + Vector3::<f32>::new(dx            , dy            , dz);
        let p3 = self.min + Vector3::<f32>::new(dx            , 0.0           , dz);
        let p4 = self.min;
        let p5 = self.min + Vector3::<f32>::new(0.0           , dy            , 0.0);
        let p6 = self.min + Vector3::<f32>::new(dx            , dy            , 0.0);
        let p7 = self.min + Vector3::<f32>::new(dx            , 0.0           , 0.0);

        let mut result: Vec<f32> = Vec::new();

        result.push(p0.x); result.push(p0.y); result.push(p0.z); result.push(1.0); 
        result.push(p1.x); result.push(p1.y); result.push(p1.z); result.push(1.0); 

        result.push(p1.x); result.push(p1.y); result.push(p1.z); result.push(1.0); 
        result.push(p2.x); result.push(p2.y); result.push(p2.z); result.push(1.0); 

        result.push(p2.x); result.push(p2.y); result.push(p2.z); result.push(1.0); 
        result.push(p3.x); result.push(p3.y); result.push(p3.z); result.push(1.0); 

        result.push(p0.x); result.push(p0.y); result.push(p0.z); result.push(1.0); 
        result.push(p3.x); result.push(p3.y); result.push(p3.z); result.push(1.0); 

        result.push(p4.x); result.push(p4.y); result.push(p4.z); result.push(1.0); 
        result.push(p5.x); result.push(p5.y); result.push(p5.z); result.push(1.0); 

        result.push(p5.x); result.push(p5.y); result.push(p5.z); result.push(1.0); 
        result.push(p6.x); result.push(p6.y); result.push(p6.z); result.push(1.0); 

        result.push(p6.x); result.push(p6.y); result.push(p6.z); result.push(1.0); 
        result.push(p7.x); result.push(p7.y); result.push(p7.z); result.push(1.0); 

        result.push(p4.x); result.push(p4.y); result.push(p4.z); result.push(1.0); 
        result.push(p7.x); result.push(p7.y); result.push(p7.z); result.push(1.0); 

        result.push(p1.x); result.push(p1.y); result.push(p1.z); result.push(1.0); 
        result.push(p5.x); result.push(p5.y); result.push(p5.z); result.push(1.0); 

        result.push(p2.x); result.push(p2.y); result.push(p2.z); result.push(1.0); 
        result.push(p6.x); result.push(p6.y); result.push(p6.z); result.push(1.0); 

        result.push(p0.x); result.push(p0.y); result.push(p0.z); result.push(1.0); 
        result.push(p4.x); result.push(p4.y); result.push(p4.z); result.push(1.0); 

        result.push(p3.x); result.push(p3.y); result.push(p3.z); result.push(1.0); 
        result.push(p7.x); result.push(p7.y); result.push(p7.z); result.push(1.0); 

        result
    }
}

/// Return min vector from a and b components.
fn min_vec(a: &Vector3<f32>, b: &Vector3<f32>) -> Vector3<f32> {
    let result = Vector3::<f32>::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z));
    result
}

/// Return max vector from a and b components.
fn max_vec(a: &Vector3<f32>, b: &Vector3<f32>) -> Vector3<f32> {
    let result = Vector3::<f32>::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z));
    result
}

/// Compute barycentric coordinates (u, v, w) for point p with respect to triangle (a, b, c)
pub fn barycentric_cooordinates(a: &Vector3<f32>, b: &Vector3<f32>, c: &Vector3<f32>, p: &Vector3<f32>) -> Vector3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    let result = Vector3::<f32>::new(u,v,w);
    result
}

pub struct Ray {
    origin: Vector3<f32>,
    dir: Vector3<f32>,
}
