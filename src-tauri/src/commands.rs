use std::path::{Path, PathBuf};
use tauri::AppHandle;

use crate::utils::{download_file, unzip_file};

#[cfg(target_os = "windows")]
fn has_command_in_path(command: &str) -> bool {
    std::process::Command::new("where")
        .arg(command)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn find_binary_from_where(command: &str) -> Option<PathBuf> {
    let output = std::process::Command::new("where")
        .arg(command)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let candidate = PathBuf::from(trimmed);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn find_binary_in_winget_packages(binary_name: &str) -> Option<PathBuf> {
    let local_app_data = std::env::var("LOCALAPPDATA").ok()?;
    let packages_root = Path::new(&local_app_data)
        .join("Microsoft")
        .join("WinGet")
        .join("Packages");

    let package_dirs = std::fs::read_dir(packages_root).ok()?;
    for package_dir in package_dirs.flatten() {
        let package_path = package_dir.path();
        if !package_path.is_dir() {
            continue;
        }

        // 兼容常见结构:
        // 1) <pkg>\<version>\bin\ffmpeg.exe
        // 2) <pkg>\bin\ffmpeg.exe
        // 3) <pkg>\<version>\ffmpeg.exe
        let mut direct_candidates = vec![
            package_path.join("bin").join(binary_name),
            package_path.join(binary_name),
        ];

        if let Ok(version_dirs) = std::fs::read_dir(&package_path) {
            for version_dir in version_dirs.flatten() {
                let version_path = version_dir.path();
                if !version_path.is_dir() {
                    continue;
                }
                direct_candidates.push(version_path.join("bin").join(binary_name));
                direct_candidates.push(version_path.join(binary_name));
            }
        }

        for candidate in direct_candidates {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    None
}

#[cfg(target_os = "windows")]
fn find_ffmpeg_binary() -> Option<PathBuf> {
    find_binary_from_where("ffmpeg")
        .or_else(|| find_binary_in_winget_packages("ffmpeg.exe"))
}

#[cfg(target_os = "windows")]
fn find_ffprobe_binary(ffmpeg_path: &Path) -> Option<PathBuf> {
    if let Some(parent) = ffmpeg_path.parent() {
        let sibling = parent.join("ffprobe.exe");
        if sibling.exists() {
            return Some(sibling);
        }
    }
    find_binary_from_where("ffprobe")
        .or_else(|| find_binary_in_winget_packages("ffprobe.exe"))
}

#[cfg(target_os = "windows")]
fn sync_ffmpeg_binaries_to_target(target_dir: &Path) -> Result<Vec<String>, String> {
    let ffmpeg_src = find_ffmpeg_binary().ok_or_else(|| "ffmpeg-not-found-after-install".to_string())?;
    let ffprobe_src = find_ffprobe_binary(&ffmpeg_src);

    let mut copied = Vec::new();

    let ffmpeg_dst = target_dir.join("ffmpeg.exe");
    std::fs::copy(&ffmpeg_src, &ffmpeg_dst).map_err(|e| {
        format!(
            "failed-to-copy-ffmpeg {} -> {}: {}",
            ffmpeg_src.to_string_lossy(),
            ffmpeg_dst.to_string_lossy(),
            e
        )
    })?;
    copied.push("ffmpeg.exe".to_string());

    if let Some(src) = ffprobe_src {
        let ffprobe_dst = target_dir.join("ffprobe.exe");
        std::fs::copy(&src, &ffprobe_dst).map_err(|e| {
            format!(
                "failed-to-copy-ffprobe {} -> {}: {}",
                src.to_string_lossy(),
                ffprobe_dst.to_string_lossy(),
                e
            )
        })?;
        copied.push("ffprobe.exe".to_string());
    }

    Ok(copied)
}

#[cfg(target_os = "windows")]
fn ensure_ffmpeg_available() -> Result<bool, String> {
    if find_ffmpeg_binary().is_some() {
        return Ok(false);
    }

    if !has_command_in_path("winget") {
        return Err("winget-not-found".to_string());
    }

    let package_ids = ["Gyan.FFmpeg", "FFmpeg.FFmpeg"];
    let mut last_error = String::new();

    for package_id in package_ids {
        let status = std::process::Command::new("winget")
            .args([
                "install",
                "--id",
                package_id,
                "--exact",
                "--silent",
                "--accept-package-agreements",
                "--accept-source-agreements",
            ])
            .status()
            .map_err(|e| format!("failed-to-run-winget: {}", e))?;

        if status.success() && find_ffmpeg_binary().is_some() {
            return Ok(true);
        }

        last_error = format!("winget-install-failed-for-{}", package_id);
    }

    Err(if last_error.is_empty() {
        "winget-install-failed".to_string()
    } else {
        last_error
    })
}

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

        let ffmpeg_installed = match ensure_ffmpeg_available() {
            Ok(installed) => installed,
            Err(e) => {
                return Err(format!("ffmpeg missing and auto install failed: {}", e));
            }
        };

        if ffmpeg_installed {
            patched.push("ffmpeg".to_string());
        }

        match sync_ffmpeg_binaries_to_target(&target) {
            Ok(copied_bins) => {
                patched.extend(copied_bins);
            }
            Err(e) => {
                return Err(format!("ffmpeg available but failed to sync binaries: {}", e));
            }
        }

        Ok(patched)
    }

    #[cfg(not(target_os = "windows"))]
    {
        let _ = target_dir;
        Ok(vec![])
    }
}
