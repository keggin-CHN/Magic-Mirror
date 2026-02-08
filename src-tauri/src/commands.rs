use std::path::{Path, PathBuf};
use tauri::AppHandle;

use crate::utils::{download_file, unzip_file};

#[tauri::command]
pub fn file_exists(path: String) -> bool {
    Path::new(&path).exists()
}

#[tauri::command]
pub async fn download_and_unzip(
    app: AppHandle,
    url: String,
    target_dir: String,
) -> Result<(), String> {
    let temp_dir = std::env::temp_dir().to_string_lossy().to_string();

    let temp_path = download_file(&app, &url, &temp_dir).await?;

    unzip_file(&app, &temp_path, &target_dir).await?;

    if let Err(e) = std::fs::remove_file(&temp_path) {
        return Err(format!("Failed to remove temp file: {}", e));
    }

    Ok(())
}

#[tauri::command]
pub fn repair_server_runtime(target_dir: String) -> Result<Vec<String>, String> {
    #[cfg(target_os = "windows")]
    {
        let target = PathBuf::from(target_dir);
        if !target.exists() {
            return Ok(vec![]);
        }

        let system_root = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".to_string());
        let system32 = Path::new(&system_root).join("System32");
        let runtime_dlls = [
            "vcruntime140.dll",
            "vcruntime140_1.dll",
            "msvcp140.dll",
            "msvcp140_1.dll",
            "msvcp140_2.dll",
            "vcomp140.dll",
        ];

        let mut patched = Vec::new();
        for dll in runtime_dlls {
            let src = system32.join(dll);
            let dst = target.join(dll);
            if !src.exists() {
                continue;
            }
            std::fs::copy(&src, &dst).map_err(|e| {
                format!(
                    "Failed to patch runtime dll {} -> {}: {}",
                    src.to_string_lossy(),
                    dst.to_string_lossy(),
                    e
                )
            })?;
            patched.push(dll.to_string());
        }

        Ok(patched)
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = target_dir;
        Ok(vec![])
    }
}
