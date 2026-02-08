mod commands;
mod utils;

use commands::{download_and_unzip, file_exists, repair_server_runtime};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_os::init())
        .invoke_handler(tauri::generate_handler![
            file_exists,
            download_and_unzip,
            repair_server_runtime
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
