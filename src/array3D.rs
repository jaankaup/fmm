use crate::bvh::{BBox}; 
use cgmath::{Vector3};

#[derive(Copy, Clone, Debug)]
pub struct CellIndex {
    pub i: u32,
    pub j: u32,
    pub k: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct Coordinate {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

//type FmmDomain = Array3D<u32, f32, fmm:Cell>; 

pub struct Array3D<T> {
    pub dimension: (std::ops::Range<u32>, std::ops::Range<u32>, std::ops::Range<u32>),
    pub base_position: Coordinate,
    pub grid_length: f32,
    pub data: Vec<T>,
    pub capacity: usize,
}

impl<T: Copy> Array3D<T> {

    pub fn new(dimension: (u32, u32, u32), base_position: Coordinate, grid_length: f32) -> Self {
        assert!(grid_length > 0.0, "Array3D.new: grid_length {} > 0.0", grid_length);

        let data: Vec<T> = Vec::with_capacity((dimension.0 as usize) * (dimension.1 as usize) * (dimension.2 as usize));
        let capacity = dimension.0 * dimension.1 * dimension.2;

        Self {
            dimension: (0..dimension.0, 0..dimension.1, 0..dimension.2),
            base_position: base_position,
            grid_length: grid_length,
            data: data,
            capacity: capacity as usize,
        }
    }

    /// Map given cell_index to a scaled and positioned x,y,z koordinates.
    pub fn map_ijk_xyz(&self, cell_index: &CellIndex) -> Option<Coordinate> {
         
        //self.validate_cell_index(&cell_index).ok()?;

        let x = cell_index.i as f32 * self.grid_length + self.base_position.x;
        let y = cell_index.j as f32 * self.grid_length + self.base_position.y;
        let z = cell_index.k as f32 * self.grid_length + self.base_position.z;

        Some(Coordinate {
            x: x,
            y: y,
            z: z,
        })
    }

    /// Map given coordinate to CellIndex.
    pub fn map_xyz_ijk(&self, coordinate: &Coordinate, ceil: bool) -> Option<CellIndex> {
         
        // Move and scale coordinate to match CellIndex space.
        let x = (((coordinate.x - self.base_position.x) / self.grid_length) + 0.01) as f32;
        let y = (((coordinate.y - self.base_position.y) / self.grid_length) + 0.01) as f32;
        let z = (((coordinate.z - self.base_position.z) / self.grid_length) + 0.01) as f32;

        assert!(x >= 0.0 && y >= 0.0 && z >= 0.0, format!("map_xyz_ijk :: {} >= 0.0 && {} >= 0.0 && {} >= 0.0", x, y, z)); 

        let i = x as u32;
        let j = y as u32;
        let k = z as u32;

        let cell_index =
            if !ceil {
                CellIndex { i: x as u32, j: y as u32, k: z as u32 }
            }
            else {
                CellIndex { i: x.ceil() as u32, j: y.ceil() as u32, k: z.ceil() as u32 }
            };

        self.validate_cell_index(&cell_index).ok()?;

        Some(cell_index)
    }

    /// Boundary check for given cell index.
    pub fn validate_cell_index(&self, cell_index: &CellIndex) -> Result<(), String> {
        if cell_index.i >= self.dimension.0.end ||
           cell_index.j >= self.dimension.1.end ||
           cell_index.k >= self.dimension.2.end  {
            let range = format!("[{}, {}, {}] - [{}, {}, {}]", 
                self.dimension.0.start,
                self.dimension.1.start,
                self.dimension.2.start,
                self.dimension.0.end,
                self.dimension.1.end,
                self.dimension.2.end
            );
            Err(format!("validate_cell_index :: cell index out of bounds. CellIndex(i: {}, j: {}, k: {}) not in range {}.", 
                    cell_index.i, cell_index.j, cell_index.k, range))
           }
           else {
                Ok(())
           }
    }

    pub fn cell_index_to_array_index(&self, cell_index: CellIndex) -> Option<u32> {
        
        self.validate_cell_index(&cell_index).ok()?;

        let x_range = self.dimension.0.end; 
        let y_range = self.dimension.1.end; 
        let index = cell_index.i + cell_index.j * x_range + cell_index.k * x_range * y_range;

        assert!(index < self.capacity as u32,
            format!("cell_index_to_array(CellIndex {{ i:{}, j:{}, k:{}}}) :: {} (index) > {}.",
                cell_index.i, cell_index.j, cell_index.k, index, self.capacity));

        Some(index)
    }

    pub fn array_index_to_cell_index(&self, index: u32) -> Option<CellIndex> {
        
        if index < self.capacity as u32 {
            let i = index % self.dimension.0.end;
            let j = (index  / self.dimension.0.end ) % self.dimension.1.end;
            let k = index / (self.dimension.0.end * self.dimension.1.end);
            Some( CellIndex{i: i, j:j, k:k} )
        }
        else {
            None
        }
    }

    pub fn get_data(&self, cell_index: CellIndex) -> Option<T> {
        
        assert!(self.capacity == self.data.len(),
            format!("Please initialize the data before accessing it. self.capacity != self.data.len(). {} != {}", self.capacity, self.data.len()));
        if let Some(index) = self.cell_index_to_array_index(cell_index) {
            Some(self.data[index as usize])
        }
        else {
            None
        }
    }

    pub fn set_data(&mut self, cell_index: CellIndex, data: T) -> Result<(), &'static str> {
         
        assert!(self.capacity == self.data.len(),
            format!("Please initialize the data before accessing it. self.capacity != self.data.len(). {} != {}", self.capacity, self.data.len()));

        if let Some(index) = self.cell_index_to_array_index(cell_index) {
            self.data[index as usize] = data;
            Ok(())
        }
        else {
            Err("set_data :: failed to add data. Do some bound checkin for cell_index.")
        }
    }

    pub fn set_data_index(&mut self, index: u32, data: T) -> Result<(), &'static str> {
         
        assert!(self.capacity == self.data.len(),
            format!("Please initialize the data before accessing it. self.capacity != self.data.len(). {} != {}", self.capacity, self.data.len()));

        if index < self.capacity as u32 {
            self.data[index as usize] = data;
            Ok(())
        }
        else {
            Err("set_data_index :: failed to add data. Do some bound checking for cell_index.")
        }
    }

    pub fn get_data_index(&mut self, index: u32) -> Option<T> {
         
        assert!(self.capacity == self.data.len(),
            format!("Please initialize the data before accessing it. self.capacity != self.data.len(). {} != {}", self.capacity, self.data.len()));

        if index < self.data.len() as u32 {
            Some(self.data[index as usize])
        }
        else {
            None
        }
    }

    /// Calculate all the indice ranges that include aabb. If aabb is outside the array3D
    /// , return None.
    pub fn aabb_to_range(&self, aabb: &BBox) -> Option<(std::ops::Range<u32>, std::ops::Range<u32>, std::ops::Range<u32>)> {

        let mut min_x = 0; 
        let mut min_y = 0; 
        let mut min_z = 0; 
        let mut max_x = 0; 
        let mut max_y = 0; 
        let mut max_z = 0; 

        /// Check if aabb can be extended.   

        let min = self.map_xyz_ijk(&Coordinate { x: aabb.min.x, y: aabb.min.y, z: aabb.min.z}, false);

        // Is the aabb min inside bounds?
        match min {
            Some(cell_index) => {
                min_x = cell_index.i;        
                min_y = cell_index.j;        
                min_z = cell_index.k;
            }
            None => {
                return None;
            }
        }

        let max = self.map_xyz_ijk(&Coordinate { x: aabb.max.x, y: aabb.max.y, z: aabb.max.z}, true);

        // Is the aabb max inside bounds?
        match max {
            Some(cell_index) => {
                max_x = cell_index.i;
                max_y = cell_index.j;
                max_z = cell_index.k; 
            }
            None => {
                return None;
            }
        }
        // Should the ranges be checked?
        Some((min_x..max_x, min_y..max_y, min_z..max_z))
    }

    /// Get all surrounded neigbors from given cell index (max 6 neighbors) Accept only neighbor cell indices which 
    /// are inside bounds. 
    pub fn get_neigbors_from_ijk(&self, cell_index: CellIndex) -> Vec<CellIndex> {
        
        if let Err(s) = self.validate_cell_index(&cell_index) {
            panic!(format!("get_neighbor_from_ijk: {}", s));
        }

        let mut neighbor_indices = Vec::new();   
        neighbor_indices.push(CellIndex { i: cell_index.i+1, j: cell_index.j  , k: cell_index.k }); 
        neighbor_indices.push(CellIndex { i: cell_index.i  , j: cell_index.j+1, k: cell_index.k }); 
        neighbor_indices.push(CellIndex { i: cell_index.i  , j: cell_index.j  , k: cell_index.k+1} ); 

        if cell_index.i != 0 { neighbor_indices.push(CellIndex { i: cell_index.i-1, j: cell_index.j  , k: cell_index.k   }) }; 
        if cell_index.j != 0 { neighbor_indices.push(CellIndex { i: cell_index.i  , j: cell_index.j-1, k: cell_index.k   }) }; 
        if cell_index.k != 0 { neighbor_indices.push(CellIndex { i: cell_index.i  , j: cell_index.j  , k: cell_index.k-1 }) }; 

        let mut result = Vec::new();

        for c in neighbor_indices.iter() {
            if let Ok(()) = self.validate_cell_index(&c) {
                // println!("Validated index ({}, {}, {})", cell_index.i, cell_index.j, cell_index.k);
                result.push(c.clone());
            }
        }

        result
    }
}

#[cfg(test)]
mod test_fmm {

    use super::*;

    #[test]
    #[should_panic]
    fn test_array3D_creation01() {
        let array: Array3D<f32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, -0.1);
    }

    #[test]
    fn test_array_map_coordinate01() {
        let array: Array3D<f32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        for i in array.dimension.0.clone() {
        for j in array.dimension.1.clone() {
        for k in array.dimension.2.clone() {
            let cell_index = CellIndex {i: i, j:j, k:k};
            let xyz_coordinate = array.map_ijk_xyz(&cell_index).unwrap();
            let inverted_cell_index = array.map_xyz_ijk(&xyz_coordinate, false).unwrap();
            assert!(cell_index.i == inverted_cell_index.i &&
                    cell_index.j == inverted_cell_index.j &&
                    cell_index.k == inverted_cell_index.k,
                    "An error occurred: {}",
                    format!("Original CellIndex {{i: {}, j: {}, k{}}} != Inverted CellIndex {{i: {}, j: {}, k: {}}}",
                        cell_index.i, cell_index.j, cell_index.k, inverted_cell_index.i, inverted_cell_index.j, inverted_cell_index.k)); 
        }}};
    }

    #[test]
    fn test_array_map_coordinate02() {
        let array: Array3D<f32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        let cell_index = CellIndex {i: 2, j:3, k:4};
        let cell_should_be = CellIndex {i: 3, j:4, k:5};
        let xyz_coordinate = array.map_ijk_xyz(&cell_index).unwrap();
        let inverted_cell_index = array.map_xyz_ijk(&xyz_coordinate, true).unwrap();
        assert!(inverted_cell_index.i == cell_should_be.i &&
                inverted_cell_index.j == cell_should_be.j &&
                inverted_cell_index.k == cell_should_be.k,
                "An error occurred: {}",
                format!("Inverted CellIndex {{i: {}, j: {}, k{}}} != Should be CellIndex {{i: {}, j: {}, k: {}}}",
                    inverted_cell_index.i, inverted_cell_index.j, inverted_cell_index.k, cell_should_be.i, cell_should_be.j, cell_should_be.k)); 
    }

    #[test]
    fn test_array_01() {
        let array: Array3D<f32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        for i in array.dimension.0.clone() {
        for j in array.dimension.1.clone() {
        for k in array.dimension.2.clone() {
            let original_cell_index = CellIndex {i: i, j:j, k:k};
            let index = array.cell_index_to_array_index(original_cell_index).unwrap();
            let inverted_cell_index = array.array_index_to_cell_index(index).unwrap();
            assert!(inverted_cell_index.i == original_cell_index.i &&
                    inverted_cell_index.j == original_cell_index.j &&
                    inverted_cell_index.k == original_cell_index.k,
                    "An error occurred: {}",
                    format!("Inverted CellIndex {{i: {}, j: {}, k{}}} != Original CellIndex {{i: {}, j: {}, k: {}}}",
                        inverted_cell_index.i,
                        inverted_cell_index.j,
                        inverted_cell_index.k,
                        original_cell_index.i,
                        original_cell_index.j,
                        original_cell_index.k)); 
        }}};
    }

    #[test]
    fn test_array_02() {
        let mut array: Array3D<u32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        array.data = vec![0 ; array.capacity];
        for i in array.dimension.0.clone() {
        for j in array.dimension.1.clone() {
        for k in array.dimension.2.clone() {
            let cell_index = CellIndex {i: i, j:j, k:k};
            let index = array.cell_index_to_array_index(cell_index).unwrap();
            let value_a = i * j * k;
            let value_b = i * j * k + 1;

            array.set_data(cell_index, value_a).unwrap();
            let result_a = array.get_data(cell_index).unwrap();

            assert!(value_a == result_a, format!("value_a != result_a {} != {}", value_a, result_a)); 

            array.set_data_index(index, value_b).unwrap();
            let result_b = array.get_data_index(index).unwrap();

            assert!(value_b == result_b, format!("value_b != result_b {} != {}", value_b, result_b)); 

        }}};
    }

    #[test]
    #[should_panic]
    fn test_array_03() {
        let mut array: Array3D<u32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        let cell_index = CellIndex {i: 3, j:2, k:1};

        array.set_data(cell_index, 123).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_array_04() {
        let mut array: Array3D<u32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        array.set_data_index(100, 123).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_array_05() {
        let mut array: Array3D<u32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        let cell_index = CellIndex {i: 3, j:2, k:1};
        array.get_data(cell_index).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_array_06() {
        let mut array: Array3D<u32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        array.get_data_index(100).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_array_07() {
        let mut array: Array3D<u32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        array.data = vec![0 ; array.capacity-3];
        let cell_index = CellIndex {i: 3, j:2, k:1};
        array.set_data(cell_index, 123).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_array_08() {
        let mut array: Array3D<u32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        array.data = vec![0 ; array.capacity];
        let cell_index = CellIndex {i: 10, j:2, k:1};
        array.set_data(cell_index, 123).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_array_09() {
        let mut array: Array3D<u32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        array.get_data_index(array.capacity as u32).unwrap();
    }

    #[test]
    fn test_array_10() {
        let mut array: Array3D<u32> = Array3D::new((10, 10, 10), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        let cell_index = array.map_xyz_ijk(&Coordinate {x: 5.11, y: 5.21, z: 5.11 }, false).unwrap();
        assert!(cell_index.i == 1 && cell_index.j == 2 && cell_index.k == 1); 

        let cell_index2 = array.map_xyz_ijk(&Coordinate {x: 5.11, y: 5.21, z: 5.11 }, true).unwrap();
        assert!(cell_index2.i == 2 && cell_index2.j == 3 && cell_index2.k == 2); 

        let cell_index3 = array.map_xyz_ijk(&Coordinate {x: 5.1, y: 5.2, z: 5.1 }, false).unwrap();
        assert!(cell_index3.i == 1 && cell_index3.j == 2 && cell_index3.k == 1); 

        let cell_index4 = array.map_xyz_ijk(&Coordinate {x: 5.1, y: 5.2, z: 5.1 }, true).unwrap();
        assert!(cell_index4.i == 2 && cell_index4.j == 3 && cell_index4.k == 2); 
    }

    #[test]
    fn test_array_11() {
        let mut array: Array3D<u32> = Array3D::new((23, 23, 23), Coordinate {x: 5.0, y: 5.0, z: 5.0 }, 0.1);
        let aabb = BBox::create_from_line(
            &Vector3::<f32>::new(5.23, 6.11, 6.50), 
            &Vector3::<f32>::new(6.00, 6.90, 7.11)
        );

        let range = array.aabb_to_range(&aabb).unwrap();
        assert!(
            range.0.start == 2  && 
            range.1.start == 11 && 
            range.2.start == 15 && 
            range.0.end   == 11 && 
            range.1.end   == 20 && 
            range.2.end   == 22
        );
    }
}
