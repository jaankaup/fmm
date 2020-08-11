use crate::misc::Convert2Vec;
use std::borrow::Cow::Borrowed;
use bytemuck::Pod;
use wgpu::util::DeviceExt;
use std::mem;

/// Buffer.
pub struct Buffer {
    pub buffer: wgpu::Buffer,
    pub capacity: usize,
    pub capacity_used: Option<usize>,
    pub label: Option<String>,
}

impl Buffer {

    pub fn create_buffer_from_data<T: Pod>(
        device: &wgpu::Device,
        t: &[T],
        usage: wgpu::BufferUsage,
        label: Option<String>)
    -> Self {
         
        let buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&t),
                usage: usage,
            }
        );
        let capacity = mem::size_of::<T>() * t.len();
        let capacity_used = Some(capacity);
        Self {
            buffer,
            capacity, 
            capacity_used, 
            label,
        }
    }

    pub fn create_buffer(device: &wgpu::Device, capacity: u64, usage: wgpu::BufferUsage, label: Option<std::borrow::Cow<&str>>) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(Borrowed("blah")),
            size: capacity,
            usage: usage,
            mapped_at_creation: false,
        });

        let capacity_used = Some(0);
        let capacity = capacity as usize; // TODO: fix this later.
        let label = None; // TODO: fix this later.

        Self {
            buffer,
            capacity,
            capacity_used,
            label,
        }
    }
    
    /// Method for copying the content of the buffer into a vector.
    pub async fn to_vec<T: Convert2Vec>(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<T> {

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: self.capacity as u64, 
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.capacity as wgpu::BufferAddress);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        //let buffer_future = staging_buffer.map_async(wgpu::MapMode::Read, 0, wgt::BufferSize::WHOLE);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);

        let res: Vec<T>;

        buffer_future.await.expect("failed"); 
        let data = buffer_slice.get_mapped_range();
        res = Convert2Vec::convert(&data);
        res
    }
}
