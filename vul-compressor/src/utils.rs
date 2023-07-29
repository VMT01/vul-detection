use std::{
    collections::{hash_map::Entry, HashMap},
    fs::{self, File, OpenOptions},
    io::{self, BufRead, Seek, SeekFrom, Write},
    path::PathBuf,
};

use crate::{
    constants::{INPUT, OUTPUT},
    log::log,
    structs::Data,
};

/* Folder interact */
pub fn read_dir() -> (Vec<PathBuf>, usize) {
    match fs::read_dir(INPUT) {
        Ok(files) => {
            let mut files: Vec<PathBuf> = files.map(|file| file.unwrap().path()).collect();
            let count = files.len();
            files.sort();
            (files, count)
        }
        Err(_) => (vec![], 0),
    }
}

/* File interact */
pub fn read_file(filename: &PathBuf) -> io::Result<io::Lines<io::BufReader<File>>> {
    match File::open(filename) {
        Ok(file) => {
            log(format!("Open {} success", filename.display()), true);
            Ok(io::BufReader::new(file).lines())
        }
        Err(err) => {
            log(
                format!("Open {} error: {:?}", filename.display(), err),
                false,
            );
            Err(err)
        }
    }
}

pub fn append_file(file: &mut File, message: String) {
    match file.write_all(message.as_bytes()) {
        Ok(_) => {}
        Err(_) => {}
    }
}

pub fn edit_file(file: &mut File, message: String, offset: u64) {
    file.seek(SeekFrom::Start(offset)).unwrap();
    match file.write(message.as_bytes()) {
        Ok(_) => {}
        Err(_) => {}
    }
}

/* Logic interact */
pub fn handle_lines(
    datas: &mut HashMap<String, Data>,
    lines: io::Lines<io::BufReader<File>>,
    line_counter: &mut u32,
    index: usize,
    vuls_count: &usize,
    global_offset: &mut u64,
) {
    for (idx, line) in lines.skip(1).enumerate() {
        match line {
            Ok(line) => {
                log(format!("Reading line {}", idx), true);
                let data: Vec<_> = line.split(",").collect();

                match datas.entry(data[0].to_string()) {
                    Entry::Occupied(mut o) => {
                        let o = o.get_mut();
                        o.add_new_label(index);

                        let message = format!(
                            "Found {} duplicate ({}/{})",
                            data[0].to_string(),
                            o.index,
                            *line_counter
                        );
                        println!("{message}");
                        log(message, true);

                        match OpenOptions::new().write(true).open(OUTPUT) {
                            Ok(mut file) => {
                                let message = o.get_label(*vuls_count);
                                edit_file(&mut file, message, o.offset);
                                log(format!("Edit line {} success", idx), true);
                            }
                            Err(_) => log(format!("Edit line {} failed", idx), false),
                        }
                    }
                    Entry::Vacant(v) => {
                        *line_counter += 1;

                        // Len of address and bytecode, plus 2 comma
                        *global_offset += (data[0].len() as u64) + (data[1].len() as u64) + 2;
                        let v = v.insert(Data::new(*line_counter, index, *global_offset));

                        // Len of label plus endl
                        *global_offset += (*vuls_count as u64) + 1;

                        match OpenOptions::new().append(true).open(OUTPUT) {
                            Ok(mut file) => {
                                let message = format!(
                                    "{},{},{}\n",
                                    data[0].to_string(),
                                    data[1].to_string(),
                                    v.get_label(*vuls_count)
                                );
                                append_file(&mut file, message);
                                log(format!("Append line {} success", idx), true);
                            }
                            Err(_) => log(format!("Append line {} failed", idx), false),
                        }
                    }
                }
            }
            Err(_) => {
                log(format!("Reading line {}", idx), false);
            }
        }
    }
}
