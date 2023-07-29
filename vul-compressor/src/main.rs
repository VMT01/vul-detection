mod constants;
mod log;
mod structs;
mod utils;

use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    process::exit,
};

use constants::{LOG, OUTPUT};
use log::log;
use structs::Data;
use utils::{append_file, handle_lines, read_dir, read_file};

fn init() {
    // First we must remove old Compressed.csv and log.txt if any
    match fs::remove_file(OUTPUT) {
        Ok(_) => log(format!("Remove {} success", OUTPUT), true),
        Err(_) => log(format!("{} not found", OUTPUT), false),
    }

    match fs::remove_file(LOG) {
        Ok(_) => log(format!("Remove {} success", LOG), true),
        Err(_) => log(format!("{} not found", LOG), false),
    }

    // Create file log
    match File::create(LOG) {
        Ok(_) => log(format!("Create {} success", LOG), true),
        Err(_) => {
            println!("Create {} failed", LOG);
            exit(-1)
        }
    }

    // Then we create new Compressed.csv with header
    match OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(OUTPUT)
    {
        Ok(mut file) => {
            log(format!("Create {} success", OUTPUT), true);
            append_file(&mut file, "ADDRESS,BYTECODE,LABEL\n".to_string());
        }
        Err(_) => {
            log(format!("Create {} failed", OUTPUT), false);
            exit(-1)
        }
    }
}

fn main() {
    init();
    let (vuls, vuls_count) = read_dir();
    let mut datas: HashMap<String, Data> = HashMap::new();
    let mut line_counter: u32 = 0;
    let mut global_offset: u64 = 23;

    // Loop for every files in input directory
    for (vul_index, vul_path) in vuls.into_iter().enumerate() {
        match read_file(&vul_path) {
            Ok(lines) => handle_lines(
                &mut datas,
                lines,
                &mut line_counter,
                vul_index,
                &vuls_count,
                &mut global_offset,
            ),
            Err(_) => {}
        };
    }
}
