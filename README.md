# Hệ thống phát hiện lỗ hổng bảo mật của hợp đồng thông minh sử dụng học sâu

Toàn bộ mã nguồn có thể tham khảo tại `https://drive.google.com/drive/folders/1Ad5yAkH6Htv1PhSpnXXXKel0MSRz8f0h`

## Sol-compiler

Đây là công cụ để có thể biên dịch mã nguồn của Solidity thành bytecode và nén thành các file tương ứng

### Required

- NodeJS v14+
- Cài đặt dependencies bằng lệnh `yarn` hoặc `npm install`

### Setup

- Thực hiện sử đổi các hằng số có trong file `src/constant/constant.json`
- Thực hiện xây dựng cây thư mục sử dụng lệnh `yarn build-folder-structure`
- Thực hiện phát hiện các version của Solidity còn thiếu sử dụng lệnh `yarn detect-missing-version`
- Thêm các phiên bản còn thiếu vào file `src/constant/version-list.json`
- Cài đặt các version còn thiếu bằng lệnh `yarn install-solc`
- Thực hiện biên dịch bằng lệnh `yarn compile`

## Vul-compressor

Đây là công cụ dùng để nén bộ dữ liệu, từ các lỗ hổng rời rạc thành nhóm các lỗ hổng, có phát hiện các contract bị trùng và đánh nhãn lại

### Required

- Cargo (Rust lang)
- Cài đặt dependencies bằng lệnh `cargo install --path`

### Setup

- Thực hiện sửa đổi các hằng số có trong `src/constant.rs`
- Chạy chương trình bằng lệnh `cargo run`

## Vul-detect-model

- Thực hiện phân loại nhị phân sử dụng file `main.py` với các lỗi tương ứng.
- Thực hiện phân loại đa nhãn sử dụng file `multi-label.py` với file đã nén `Data_Cleasing.csv`
