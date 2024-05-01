use image::imageops::grayscale;
use image::io::Reader as ImageReader;
use image::ImageBuffer;
use image::Luma;
use image::Rgb;
use imageproc::filter;
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

fn grey_to_float(
    g: &ImageBuffer<Luma<u8>, Vec<u8>>,
    f: fn(f32) -> f32,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    let floats: Vec<f32> = g.iter().map(|u| f(to_float_0_1(*u))).collect();
    ImageBuffer::<Luma<f32>, _>::from_raw(g.width(), g.height(), floats).unwrap()
}

fn to_float_0_1(u: u8) -> f32 {
    u as f32 / 255.0
}

fn to_u8_0_1(f: f32) -> u8 {
    (f * 255.0).round_ties_even() as u8
}

fn to_float_n1_1(u: u8) -> f32 {
    (to_float_0_1(u) - 0.5) * 2.0
}

fn to_u8_n1_1(f: f32) -> u8 {
    to_u8_0_1((f + 1.0) / 2.0)
}

fn identity(f: f32) -> f32 {
    f
}

fn edges(image: &ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let g = grayscale(image);
    let f = grey_to_float(&g, identity);
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
        .map(|(((v1, v2), v3), v4)| to_u8_0_1(v1.max(*v2).max(*v3).max(*v4)))
        .collect();
    image::ImageBuffer::<Luma<u8>, _>::from_raw(image.width(), image.height(), b).unwrap()
}

fn binary_edge(f: f32) -> f32 {
    if f > 0.3 && f < 0.8 {
        1.0
    } else {
        0.0
    }
}

fn save_f32_luma(image: &ImageBuffer<Luma<f32>, Vec<f32>>, path: &str) {
    let eye_data: Vec<u8> = image.iter().map(|f| to_u8_0_1(*f)).collect();
    let eye = image::ImageBuffer::<Luma<u8>, _>::from_raw(image.width(), image.height(), eye_data)
        .unwrap();
    eye.save(path).unwrap();
}
fn save_f32_luma_vec(width: u32, height: u32, d: &[f32], path: &str) {
    let eye_data: Vec<u8> = d.iter().map(|f| to_u8_0_1(*f)).collect();
    let eye = image::ImageBuffer::<Luma<u8>, _>::from_raw(width, height, eye_data).unwrap();
    eye.save(path).unwrap();
}

fn convolution_pass(f: f32) -> f32 {
    if f > 0.3 {
        3.0
    } else {
        0.0
    }
}

fn threshold(f: f32) -> f32 {
    if f > 0.3 {
        1.0
    } else {
        0.0
    }
}

fn eye_convolution_from_image() -> (Vec<f32>, u32, u32) {
    let png = ImageReader::open("eye.png").unwrap().decode().unwrap();
    let g = grayscale(&png);
    let v = grey_to_float(&g, convolution_pass);
    (v.into_iter().cloned().collect(), v.width(), v.height())
}

fn detect_eyes(image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> bool {
    let g = grey_to_float(image, threshold);
    save_f32_luma(&g, "after_conversion.png");
    let s = 32 as usize;
    // let circel = create_circle_convolution(s, 0.3, 0.4, 0.05);
    // save_f32_luma_vec(s as u32, s as u32, &circel, "circle.png");
    let eye_conv = eye_convolution_from_image();
    save_f32_luma_vec(eye_conv.1, eye_conv.2, &eye_conv.0, "circle.png");
    let kernel = Kernel::<f32>::new(&eye_conv.0[..], eye_conv.1, eye_conv.2);
    let mut eye_detected = kernel.filter::<Luma<f32>, _, Luma<f32>>(&g, |channel, acc| {
        if acc > eye_conv.0.len() as f32 / 6.0 {
            *channel = 1.0;
        } else {
            dbg!(acc);
            *channel = 0.0;
        }
    });

    save_f32_luma(&eye_detected, "pre_eye_detected.png");

    let tiny_circle_conv = create_circle_convolution(64, 0.0, 0.0, 0.04);
    save_f32_luma_vec(64, 64, &tiny_circle_conv, "tiny.png");
    let kernel = Kernel::<f32>::new(&tiny_circle_conv[..], 64, 64);
    let eye_detected =
        kernel.filter::<Luma<f32>, _, Luma<f32>>(&eye_detected, |channel, acc| *channel = acc);
    save_f32_luma(&eye_detected, "eye_detected.png");
    false
}

fn create_circle_convolution(
    size: usize,
    inner_radius: f32,
    outer_radius: f32,
    inner_dot_radius: f32,
) -> Vec<f32> {
    let mut data = Vec::<f32>::with_capacity(size * size);

    for x in 0..size {
        for y in 0..size {
            let x = ((x as f32) - (size as f32) / 2.0) / (size as f32);
            let y = ((y as f32) - (size as f32) / 2.0) / (size as f32);
            let distance = ((x) * (x) + (y) * (y)).sqrt();

            if (distance >= inner_radius && distance <= outer_radius) || distance < inner_dot_radius
            {
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
    use image::{imageops::grayscale, io::Reader as ImageReader};

    #[test]
    fn convert_0_1() {
        for u in 0..=255u8 {
            let f = to_float_0_1(u);
            assert!(f >= 0.0);
            assert!(f <= 1.0);
            assert!(to_u8_0_1(f) == u);
        }
    }
    #[test]
    fn convert_n1_1() {
        for u in 0..=255u8 {
            let f = to_float_n1_1(u);
            dbg!(u);
            dbg!(f);
            if u < 128 {
                assert!(f >= -1.0);
                assert!(f <= 0.0);
            } else {
                assert!(f >= 0.0);
                assert!(f <= 1.0);
            }
            assert!(dbg!(to_u8_n1_1(f)) == u);
        }
    }

    use crate::{detect_eyes, to_float_0_1, to_float_n1_1, to_u8_0_1, to_u8_n1_1};
    #[test]
    fn eye_detection_exists_with_glasses() {
        let png = ImageReader::open("edges_eyes_glasses.png")
            .unwrap()
            .decode()
            .unwrap();
        let g = grayscale(&png);
        assert!(detect_eyes(&g));
    }
}
