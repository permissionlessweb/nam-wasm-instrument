//! Utility data structures and functions.

use alloc::collections::btree_set::BTreeSet;

use parity_wasm::elements;

/// Gather all the imported function ids.
pub fn imported_function_ids(module: &elements::Module) -> BTreeSet<u32> {
	let func_imports = module.import_count(elements::ImportCountType::Function) as u32;
	(0..func_imports).collect()
}
