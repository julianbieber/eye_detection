use image::imageops::grayscale;
use image::ImageBuffer;
use image::Luma;
use image::Rgb;
use imageproc::filter;
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};

fn main() {
    // first camera in system
    let index = CameraIndex::Index(0);
    // request the absolute highest resolution CameraFormat that can be decoded to RGB.
    let requested =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
    // make the camera
    let mut camera = Camera::new(index, requested).unwrap();
    camera.open_stream().unwrap();

    // get a frame
    let frame = camera.frame().unwrap();
    println!("Captured Single Frame of {}", frame.buffer().len());
    // decode into an ImageBuffer
    let decoded = frame.decode_image::<RgbFormat>().unwrap();
    let e = edges(&decoded);
    e
    e.save("image.png").unwrap();
    println!("Decoded Frame of {}", decoded.len());
}

fn grey_to_float(g: &ImageBuffer<Luma<u8>, Vec<u8>>) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let floats: Vec<f32> = g.iter().map(|u| *u as f32 / 255.0).collect();
    ImageBuffer::<Luma<f32>, _>::from_raw(g.width(), g.height(), floats).unwrap()
}

fn edges(image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let g = grayscale(image);
    let f = grey_to_float(&g);
    let filtered = filter::filter3x3::<Luma<f32>, f32, f32>(
        &f,
        &[1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0],
    );
    let filtered2 = filter::filter3x3::<Luma<f32>, f32, f32>(
        &f,
        &[1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0],
    );
    let filtered3 = filter::filter3x3::<Luma<f32>, f32, f32>(
        &f,
        &[0.0, 1.0, 2.0, -1.0, 0.0, 1.0, -2.0, -1.0, 0.0],
    );
    let filtered4 = filter::filter3x3::<Luma<f32>, f32, f32>(
        &f,
        &[-2.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 2.0],
    );

    let b: Vec<u8> = filtered
        .iter()
        .zip(filtered2.iter())
        .zip(filtered3.iter())
        .zip(filtered4.iter())
        .map(|(((v1, v2), v3), v4)| ((v1.max(*v2).max(*v3).max(*v4)) * 255.0) as u8)
        .collect();
    image::ImageBuffer::<Luma<u8>, _>::from_raw(image.width(), image.height(), b).unwrap()
}
