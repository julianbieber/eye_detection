use image::codecs::pnm::GraymapHeader;
use image::imageops::grayscale;
use image::GenericImageView;
use image::ImageBuffer;
use image::ImageDecoder;
use image::Luma;
use image::Rgb;
use imageproc::filter;
use imageproc::filter::filter3x3;
use imageproc::filter::Kernel;
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

fn detect_eyes(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> bool {
    let g = grey_to_float(image);
    let s = (100) as usize;
    let circel = create_circle_convolution(s, 0.55, 0.8);
    let kernel = Kernel::<f32>::new(&circel[..], s as u32, s as u32);
    use imageproc::definitions::Clamp;
    let eye_detected = kernel.filter::<Luma<f32>, _, Luma<f32>>(&g, |channel, acc| {
        *channel = <f32 as Clamp<f32>>::clamp(acc)
    });
    let eye_data: Vec<u8> = eye_detected.iter().map(|f| (f * 255.0) as u8).collect();
    let eye = image::ImageBuffer::<Luma<u8>, _>::from_raw(image.height(), image.width(), eye_data)
        .unwrap();
    eye.save("eye_detected.png").unwrap();
    false
}

fn create_circle_convolution(size: usize, inner_radius: f32, outer_radius: f32) -> Vec<f32> {
    let mut data = Vec::<f32>::with_capacity(size * size);

    for x in 0..size {
        for y in 0..size {
            let x = ((x as f32) - (size as f32) / 2.0) / (size as f32);
            let y = ((y as f32) - (size as f32) / 2.0) / (size as f32);
            let distance = ((x) * (x) + (y) * (y)).sqrt();

            if distance > inner_radius && distance < outer_radius {
                data.push(2.0);
            } else {
                data.push(-2.0)
            }
        }
    }
    data
}

#[cfg(test)]
mod test {
    use image::{imageops::grayscale, io::Reader as ImageReader, Luma};

    use crate::{create_circle_convolution, detect_eyes};
    #[test]
    fn eye_detection_exists_with_glasses() {
        let png = ImageReader::open("edges_eyes_glasses.png")
            .unwrap()
            .decode()
            .unwrap();
        let g = grayscale(&png);
        assert!(detect_eyes(&g));
    }

    #[test]
    fn create_circle_convolution_save_to_file() {
        let circel_data = create_circle_convolution(128, 0.2, 1.0);
        image::ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(128, 128, circel_data)
            .unwrap()
            .save("circle.png")
            .unwrap();
    }
}
