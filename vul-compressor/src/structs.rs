#[derive(Debug)]
pub struct Data {
    pub index: u32,
    label: Vec<usize>,
    pub offset: u64,
}

impl Data {
    pub fn new(index: u32, label: usize, offset: u64) -> Data {
        Data {
            index,
            label: vec![label],
            offset,
        }
    }

    pub fn add_new_label(&mut self, label: usize) {
        self.label.push(label);
    }

    pub fn get_label(&self, size: usize) -> String {
        format!(
            "{:0size$b}",
            self.label.iter().map(|e| 1 << e).sum::<usize>(),
            size = size
        )
    }
}
