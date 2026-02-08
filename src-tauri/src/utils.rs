use futures_util::StreamExt;
use reqwest::Client;
use std::fs::{self, create_dir_all, File};
use std::io::{copy, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tauri::AppHandle;
use tauri::Emitter;
use zip::read::ZipArchive;

pub fn timestamp() -> u128 {
    let start = SystemTime::now();
    let duration = start
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards");
    duration.as_millis()
}

pub async fn download_file(
    app: &AppHandle,
    url: &String,
    temp_dir: &String,
) -> Result<String, String> {
    let temp_path = Path::new(temp_dir).join(format!("{}.zip", timestamp()));

    let client = Client::new();
    let response = match client.get(url).send().await {
        Ok(res) => res,
        Err(e) => {
            return Err(format!("Failed to download file: {}", e));
        }
    };

    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded = 0u64;
    let mut file = match File::create(&temp_path) {
        Ok(f) => f,
        Err(e) => {
            return Err(format!("Failed to create temp file: {}", e));
        }
    };

    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                return Err(format!("Failed to read chunk: {}", e));
            }
        };
        if let Err(e) = file.write_all(&chunk) {
            return Err(format!("Failed to write chunk to file: {}", e));
        }
        downloaded += chunk.len() as u64;
        if total_size > 0 {
            let progress = downloaded as f64 / total_size as f64 * 100.0;
            app.emit("download-progress", progress).unwrap_or_default();
        }
    }

    Ok(temp_path.to_string_lossy().to_string())
}

pub async fn unzip_file(
    app: &AppHandle,
    file_path: &String,
    target_dir: &String,
) -> Result<(), String> {
    let file = match File::open(file_path) {
        Ok(f) => f,
        Err(e) => {
            return Err(format!("Failed to open temp file: {}", e));
        }
    };

    let target_path = Path::new(target_dir);
    if target_path.exists() {
        if let Err(e) = fs::remove_dir_all(target_path) {
            return Err(format!("Failed to remove target directory: {}", e));
        }
    }
    if let Err(e) = create_dir_all(target_path) {
        return Err(format!("Failed to create target directory: {}", e));
    }

    let mut archive = match ZipArchive::new(file) {
        Ok(a) => a,
        Err(e) => {
            return Err(format!("Failed to read zip archive: {}", e));
        }
    };

    let total_files = archive.len();
    for i in 0..total_files {
        let mut file = match archive.by_index(i) {
            Ok(f) => f,
            Err(e) => {
                return Err(format!("Failed to read file from zip: {}", e));
            }
        };

        let outpath = match file.enclosed_name() {
            Some(path) => target_path.join(path),
            None => continue,
        };

        if file.name().ends_with('/') {
            if let Err(e) = create_dir_all(&outpath) {
                return Err(format!("Failed to create directory: {}", e));
            }
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    if let Err(e) = create_dir_all(p) {
                        return Err(format!("Failed to create parent directory: {}", e));
                    }
                }
            }

            let mut outfile = match File::create(&outpath) {
                Ok(f) => f,
                Err(e) => {
                    return Err(format!("Failed to create output file: {}", e));
                }
            };

            if let Err(e) = copy(&mut file, &mut outfile) {
                return Err(format!("Failed to copy file content: {}", e));
            }
        }

        let progress = i as f64 / total_files as f64 * 100.0;
        app.emit("unzip-progress", progress).unwrap_or_default();
    }

    Ok(())
}
