use std::fs::OpenOptions;

use crate::{constants::LOG, utils::append_file};

pub fn log(message: String, is_success: bool) {
    let message = format!("{} {}\n", if is_success { "+" } else { "- " }, message);
    match OpenOptions::new().append(true).open(LOG) {
        Ok(mut file) => append_file(&mut file, message),
        Err(_) => {}
    }
}
