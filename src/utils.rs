#[inline]
pub(crate) fn generate_file_id_str(id: usize) -> String {
    let mut buffer = itoa::Buffer::new();
    let formatted = buffer.format(id);
    formatted.to_owned()
}
