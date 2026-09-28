use super::resolve_func_type;
use alloc::vec::Vec;
use parity_wasm::elements::{self, BlockType, Type};

#[cfg(feature = "sign_ext")]
use parity_wasm::elements::SignExtInstruction;

#[cfg(feature = "bulk")]
use parity_wasm::elements::BulkInstruction;

#[cfg(feature = "simd")]
use parity_wasm::elements::SimdInstruction;

// The cost in stack items that should be charged per call of a function. This is
// is a static cost that is added to each function call. This makes sense because even
// if a function does not use any parameters or locals some stack space on the host
// machine might be consumed to hold some context.
const ACTIVATION_FRAME_COST: u32 = 2;

/// Control stack frame.
#[derive(Debug)]
struct Frame {
	/// Stack becomes polymorphic only after an instruction that
	/// never passes control further was executed.
	is_polymorphic: bool,

	/// Count of values which will be pushed after the exit
	/// from the current block.
	end_arity: u32,

	/// Count of values which should be poped upon a branch to
	/// this frame.
	///
	/// This might be diffirent from `end_arity` since branch
	/// to the loop header can't take any values.
	branch_arity: u32,

	/// Stack height before entering in the block.
	start_height: u32,
}

/// This is a compound stack that abstracts tracking height of the value stack
/// and manipulation of the control stack.
struct Stack {
	height: u32,
	control_stack: Vec<Frame>,
}

impl Stack {
	fn new() -> Stack {
		Stack { height: ACTIVATION_FRAME_COST, control_stack: Vec::new() }
	}

	/// Returns current height of the value stack.
	fn height(&self) -> u32 {
		self.height
	}

	/// Returns a reference to a frame by specified depth relative to the top of
	/// control stack.
	fn frame(&self, rel_depth: u32) -> Result<&Frame, &'static str> {
		let control_stack_height: usize = self.control_stack.len();
		let last_idx = control_stack_height.checked_sub(1).ok_or("control stack is empty")?;
		let idx = last_idx.checked_sub(rel_depth as usize).ok_or("control stack out-of-bounds")?;
		Ok(&self.control_stack[idx])
	}

	/// Mark successive instructions as unreachable.
	///
	/// This effectively makes stack polymorphic.
	fn mark_unreachable(&mut self) -> Result<(), &'static str> {
		let top_frame = self.control_stack.last_mut().ok_or("stack must be non-empty")?;
		top_frame.is_polymorphic = true;
		Ok(())
	}

	/// Push control frame into the control stack.
	fn push_frame(&mut self, frame: Frame) {
		self.control_stack.push(frame);
	}

	/// Pop control frame from the control stack.
	///
	/// Returns `Err` if the control stack is empty.
	fn pop_frame(&mut self) -> Result<Frame, &'static str> {
		self.control_stack.pop().ok_or("stack must be non-empty")
	}

	/// Truncate the height of value stack to the specified height.
	fn trunc(&mut self, new_height: u32) {
		self.height = new_height;
	}

	/// Push specified number of values into the value stack.
	///
	/// Returns `Err` if the height overflow usize value.
	fn push_values(&mut self, value_count: u32) -> Result<(), &'static str> {
		self.height = self.height.checked_add(value_count).ok_or("stack overflow")?;
		Ok(())
	}

	/// Pop specified number of values from the value stack.
	///
	/// Returns `Err` if the stack happen to be negative value after
	/// values popped.
	fn pop_values(&mut self, value_count: u32) -> Result<(), &'static str> {
		if value_count == 0 {
			return Ok(())
		}
		{
			let top_frame = self.frame(0)?;
			if self.height == top_frame.start_height {
				// It is an error to pop more values than was pushed in the current frame
				// (ie pop values pushed in the parent frame), unless the frame became
				// polymorphic.
				return if top_frame.is_polymorphic {
					Ok(())
				} else {
					return Err("trying to pop more values than pushed")
				}
			}
		}

		self.height = self.height.checked_sub(value_count).ok_or("stack underflow")?;

		Ok(())
	}
}

/// This function expects the function to be validated.
pub fn compute(func_idx: u32, module: &elements::Module) -> Result<u32, &'static str> {
	use parity_wasm::elements::Instruction::*;

	let func_section = module.function_section().ok_or("No function section")?;
	let code_section = module.code_section().ok_or("No code section")?;
	let type_section = module.type_section().ok_or("No type section")?;

	// Get a signature and a body of the specified function.
	let func_sig_idx = func_section
		.entries()
		.get(func_idx as usize)
		.ok_or("Function is not found in func section")?
		.type_ref();
	let Type::Function(func_signature) = type_section
		.types()
		.get(func_sig_idx as usize)
		.ok_or("Function is not found in func section")?;
	let body = code_section
		.bodies()
		.get(func_idx as usize)
		.ok_or("Function body for the index isn't found")?;
	let instructions = body.code();

	let mut stack = Stack::new();
	let mut max_height: u32 = 0;
	let mut pc = 0;

	// Add implicit frame for the function. Breaks to this frame and execution of
	// the last end should deal with this frame.
	let func_arity = func_signature.results().len() as u32;
	stack.push_frame(Frame {
		is_polymorphic: false,
		end_arity: func_arity,
		branch_arity: func_arity,
		start_height: 0,
	});

	loop {
		if pc >= instructions.elements().len() {
			break
		}

		// If current value stack is higher than maximal height observed so far,
		// save the new height.
		// However, we don't increase maximal value in unreachable code.
		if stack.height() > max_height && !stack.frame(0)?.is_polymorphic {
			max_height = stack.height();
		}

		let opcode = &instructions.elements()[pc];

		match opcode {
			Nop => {},
			Block(ty) | Loop(ty) | If(ty) => {
				let end_arity = u32::from(*ty != BlockType::NoResult);
				let branch_arity = if let Loop(_) = *opcode { 0 } else { end_arity };
				if let If(_) = *opcode {
					stack.pop_values(1)?;
				}
				let height = stack.height();
				stack.push_frame(Frame {
					is_polymorphic: false,
					end_arity,
					branch_arity,
					start_height: height,
				});
			},
			Else => {
				// The frame at the top should be pushed by `If`. So we leave
				// it as is.
			},
			End => {
				let frame = stack.pop_frame()?;
				stack.trunc(frame.start_height);
				stack.push_values(frame.end_arity)?;
			},
			Unreachable => {
				stack.mark_unreachable()?;
			},
			Br(target) => {
				// Pop values for the destination block result.
				let target_arity = stack.frame(*target)?.branch_arity;
				stack.pop_values(target_arity)?;

				// This instruction unconditionally transfers control to the specified block,
				// thus all instruction until the end of the current block is deemed unreachable
				stack.mark_unreachable()?;
			},
			BrIf(target) => {
				// Pop values for the destination block result.
				let target_arity = stack.frame(*target)?.branch_arity;
				stack.pop_values(target_arity)?;

				// Pop condition value.
				stack.pop_values(1)?;

				// Push values back.
				stack.push_values(target_arity)?;
			},
			BrTable(br_table_data) => {
				let arity_of_default = stack.frame(br_table_data.default)?.branch_arity;

				// Check that all jump targets have an equal arities.
				for target in &*br_table_data.table {
					let arity = stack.frame(*target)?.branch_arity;
					if arity != arity_of_default {
						return Err("Arity of all jump-targets must be equal")
					}
				}

				// Because all jump targets have an equal arities, we can just take arity of
				// the default branch.
				stack.pop_values(arity_of_default)?;

				// This instruction doesn't let control flow to go further, since the control flow
				// should take either one of branches depending on the value or the default branch.
				stack.mark_unreachable()?;
			},
			Return => {
				// Pop return values of the function. Mark successive instructions as unreachable
				// since this instruction doesn't let control flow to go further.
				stack.pop_values(func_arity)?;
				stack.mark_unreachable()?;
			},
			Call(idx) => {
				let ty = resolve_func_type(*idx, module)?;

				// Pop values for arguments of the function.
				stack.pop_values(ty.params().len() as u32)?;

				// Push result of the function execution to the stack.
				let callee_arity = ty.results().len() as u32;
				stack.push_values(callee_arity)?;
			},
			CallIndirect(x, _) => {
				let Type::Function(ty) =
					type_section.types().get(*x as usize).ok_or("Type not found")?;

				// Pop the offset into the function table.
				stack.pop_values(1)?;

				// Pop values for arguments of the function.
				stack.pop_values(ty.params().len() as u32)?;

				// Push result of the function execution to the stack.
				let callee_arity = ty.results().len() as u32;
				stack.push_values(callee_arity)?;
			},
			Drop => {
				stack.pop_values(1)?;
			},
			Select => {
				// Pop two values and one condition.
				stack.pop_values(2)?;
				stack.pop_values(1)?;

				// Push the selected value.
				stack.push_values(1)?;
			},
			GetLocal(_) => {
				stack.push_values(1)?;
			},
			SetLocal(_) => {
				stack.pop_values(1)?;
			},
			TeeLocal(_) => {
				// This instruction pops and pushes the value, so
				// effectively it doesn't modify the stack height.
				stack.pop_values(1)?;
				stack.push_values(1)?;
			},
			GetGlobal(_) => {
				stack.push_values(1)?;
			},
			SetGlobal(_) => {
				stack.pop_values(1)?;
			},
			I32Load(_, _) |
			I64Load(_, _) |
			F32Load(_, _) |
			F64Load(_, _) |
			I32Load8S(_, _) |
			I32Load8U(_, _) |
			I32Load16S(_, _) |
			I32Load16U(_, _) |
			I64Load8S(_, _) |
			I64Load8U(_, _) |
			I64Load16S(_, _) |
			I64Load16U(_, _) |
			I64Load32S(_, _) |
			I64Load32U(_, _) => {
				// These instructions pop the address and pushes the result,
				// which effictively don't modify the stack height.
				stack.pop_values(1)?;
				stack.push_values(1)?;
			},

			I32Store(_, _) |
			I64Store(_, _) |
			F32Store(_, _) |
			F64Store(_, _) |
			I32Store8(_, _) |
			I32Store16(_, _) |
			I64Store8(_, _) |
			I64Store16(_, _) |
			I64Store32(_, _) => {
				// These instructions pop the address and the value.
				stack.pop_values(2)?;
			},

			CurrentMemory(_) => {
				// Pushes current memory size
				stack.push_values(1)?;
			},
			GrowMemory(_) => {
				// Grow memory takes the value of pages to grow and pushes
				stack.pop_values(1)?;
				stack.push_values(1)?;
			},

			I32Const(_) | I64Const(_) | F32Const(_) | F64Const(_) => {
				// These instructions just push the single literal value onto the stack.
				stack.push_values(1)?;
			},

			I32Eqz | I64Eqz => {
				// These instructions pop the value and compare it against zero, and pushes
				// the result of the comparison.
				stack.pop_values(1)?;
				stack.push_values(1)?;
			},

			I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS |
			I32GeU | I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU |
			I64GeS | I64GeU | F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge | F64Eq | F64Ne |
			F64Lt | F64Gt | F64Le | F64Ge => {
				// Comparison operations take two operands and produce one result.
				stack.pop_values(2)?;
				stack.push_values(1)?;
			},

			I32Clz | I32Ctz | I32Popcnt | I64Clz | I64Ctz | I64Popcnt | F32Abs | F32Neg |
			F32Ceil | F32Floor | F32Trunc | F32Nearest | F32Sqrt | F64Abs | F64Neg | F64Ceil |
			F64Floor | F64Trunc | F64Nearest | F64Sqrt => {
				// Unary operators take one operand and produce one result.
				stack.pop_values(1)?;
				stack.push_values(1)?;
			},

			I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or |
			I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr | I64Add | I64Sub |
			I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or | I64Xor | I64Shl |
			I64ShrS | I64ShrU | I64Rotl | I64Rotr | F32Add | F32Sub | F32Mul | F32Div |
			F32Min | F32Max | F32Copysign | F64Add | F64Sub | F64Mul | F64Div | F64Min |
			F64Max | F64Copysign => {
				// Binary operators take two operands and produce one result.
				stack.pop_values(2)?;
				stack.push_values(1)?;
			},

			I32WrapI64 | I32TruncSF32 | I32TruncUF32 | I32TruncSF64 | I32TruncUF64 |
			I64ExtendSI32 | I64ExtendUI32 | I64TruncSF32 | I64TruncUF32 | I64TruncSF64 |
			I64TruncUF64 | F32ConvertSI32 | F32ConvertUI32 | F32ConvertSI64 | F32ConvertUI64 |
			F32DemoteF64 | F64ConvertSI32 | F64ConvertUI32 | F64ConvertSI64 | F64ConvertUI64 |
			F64PromoteF32 | I32ReinterpretF32 | I64ReinterpretF64 | F32ReinterpretI32 |
			F64ReinterpretI64 => {
				// Conversion operators take one value and produce one result.
				stack.pop_values(1)?;
				stack.push_values(1)?;
			},

			#[cfg(feature = "sign_ext")]
			SignExt(SignExtInstruction::I32Extend8S) |
			SignExt(SignExtInstruction::I32Extend16S) |
			SignExt(SignExtInstruction::I64Extend8S) |
			SignExt(SignExtInstruction::I64Extend16S) |
			SignExt(SignExtInstruction::I64Extend32S) => {
				stack.pop_values(1)?;
				stack.push_values(1)?;
			},

			// memory.init/copy/fill and table.init/copy take (dst, src, len).
			#[cfg(feature = "bulk")]
			Bulk(BulkInstruction::MemoryInit(_)) |
			Bulk(BulkInstruction::MemoryCopy) |
			Bulk(BulkInstruction::MemoryFill) |
			Bulk(BulkInstruction::TableInit(_)) |
			Bulk(BulkInstruction::TableCopy) => {
				stack.pop_values(3)?;
			},

			#[cfg(feature = "bulk")]
			Bulk(BulkInstruction::MemoryDrop(_)) |
			Bulk(BulkInstruction::TableDrop(_)) => {},

			#[cfg(feature = "simd")]
			Simd(ref op) => simd_stack(op, &mut stack)?,
		}
		pc += 1;
	}

	Ok(max_height)
}

#[cfg(feature = "simd")]
fn simd_stack(op: &SimdInstruction, stack: &mut Stack) -> Result<(), &'static str> {
	use SimdInstruction::*;
	match op {
		V128Const(_) |
		V128Load(_) |
		V128Load8x8S(_) |
		V128Load8x8U(_) |
		V128Load16x4S(_) |
		V128Load16x4U(_) |
		V128Load32x2S(_) |
		V128Load32x2U(_) |
		V128Load8Splat(_) |
		V128Load16Splat(_) |
		V128Load32Splat(_) |
		V128Load64Splat(_) |
		V128Load32Zero(_) |
		V128Load64Zero(_) => { stack.push_values(1)?; }
		V128Store(_) |
		V128Store8Lane(_, _) |
		V128Store16Lane(_, _) |
		V128Store32Lane(_, _) |
		V128Store64Lane(_, _) => { stack.pop_values(1)?; }
		I8x16Splat |
		I16x8Splat |
		I32x4Splat |
		I64x2Splat |
		F32x4Splat |
		F64x2Splat |
		I8x16ExtractLaneS(_) |
		I8x16ExtractLaneU(_) |
		I16x8ExtractLaneS(_) |
		I16x8ExtractLaneU(_) |
		I32x4ExtractLane(_) |
		I64x2ExtractLane(_) |
		F32x4ExtractLane(_) |
		F64x2ExtractLane(_) |
		V128Not |
		I8x16AnyTrue |
		I16x8AnyTrue |
		I32x4AnyTrue |
		I64x2AnyTrue |
		I8x16AllTrue |
		I16x8AllTrue |
		I32x4AllTrue |
		I64x2AllTrue |
		F32x4Abs |
		F64x2Abs |
		F32x4Div |
		F64x2Div |
		F32x4Sqrt |
		F64x2Sqrt |
		F32x4ConvertSI32x4 |
		F32x4ConvertUI32x4 |
		F64x2ConvertSI64x2 |
		F64x2ConvertUI64x2 |
		I32x4TruncSF32x4Sat |
		I32x4TruncUF32x4Sat |
		I64x2TruncSF64x2Sat |
		I64x2TruncUF64x2Sat |
		V128AnyTrue |
		V128Load8Lane(_, _) |
		V128Load16Lane(_, _) |
		V128Load32Lane(_, _) |
		V128Load64Lane(_, _) |
		F32x4DemoteF64x2Zero |
		F64x2PromoteLowF32x4 |
		I8x16Abs |
		I8x16Popcnt |
		I8x16Bitmask |
		F32x4Ceil |
		F32x4Floor |
		F32x4Trunc |
		F64x2Ceil |
		F64x2Floor |
		F64x2Trunc |
		I16x8ExtaddPairwiseI8x16S |
		I16x8ExtaddPairwiseI8x16U |
		I32x4ExtaddPairwiseI16x8S |
		I32x4ExtaddPairwiseI16x8U |
		I16x8Abs |
		I16x8Bitmask |
		I16x8ExtendLowI8x16S |
		I16x8ExtendHighI8x16S |
		I16x8ExtendLowI8x16U |
		I16x8ExtendHighI8x16U |
		I32x4Abs |
		I32x4Bitmask |
		I32x4ExtendLowI16x8S |
		I32x4ExtendHighI16x8S |
		I32x4ExtendLowI16x8U |
		I32x4ExtendHighI16x8U |
		I64x2Abs |
		I64x2Bitmask |
		I64x2ExtendLowI32x4S |
		I64x2ExtendHighI32x4S |
		I64x2ExtendLowI32x4U |
		I64x2ExtendHighI32x4U |
		I32x4TruncSatF64x2SZero |
		I32x4TruncSatF64x2UZero |
		F64x2ConvertLowI32x4S |
		F64x2ConvertLowI32x4U => { stack.pop_values(1)?; stack.push_values(1)?; }
		I8x16ReplaceLane(_) |
		I16x8ReplaceLane(_) |
		I32x4ReplaceLane(_) |
		I64x2ReplaceLane(_) |
		F32x4ReplaceLane(_) |
		F64x2ReplaceLane(_) |
		V8x16Shuffle(_) |
		I8x16Add |
		I16x8Add |
		I32x4Add |
		I64x2Add |
		I8x16Sub |
		I16x8Sub |
		I32x4Sub |
		I64x2Sub |
		I8x16Mul |
		I16x8Mul |
		I32x4Mul |
		I64x2Mul |
		I8x16Neg |
		I16x8Neg |
		I32x4Neg |
		I64x2Neg |
		I8x16AddSaturateS |
		I8x16AddSaturateU |
		I16x8AddSaturateS |
		I16x8AddSaturateU |
		I8x16SubSaturateS |
		I8x16SubSaturateU |
		I16x8SubSaturateS |
		I16x8SubSaturateU |
		I8x16Shl |
		I16x8Shl |
		I32x4Shl |
		I64x2Shl |
		I8x16ShrS |
		I8x16ShrU |
		I16x8ShrS |
		I16x8ShrU |
		I32x4ShrS |
		I32x4ShrU |
		I64x2ShrS |
		I64x2ShrU |
		V128And |
		V128Or |
		V128Xor |
		I8x16Eq |
		I16x8Eq |
		I32x4Eq |
		I64x2Eq |
		F32x4Eq |
		F64x2Eq |
		I8x16Ne |
		I16x8Ne |
		I32x4Ne |
		I64x2Ne |
		F32x4Ne |
		F64x2Ne |
		I8x16LtS |
		I8x16LtU |
		I16x8LtS |
		I16x8LtU |
		I32x4LtS |
		I32x4LtU |
		I64x2LtS |
		F32x4Lt |
		F64x2Lt |
		I8x16LeS |
		I8x16LeU |
		I16x8LeS |
		I16x8LeU |
		I32x4LeS |
		I32x4LeU |
		I64x2LeS |
		F32x4Le |
		F64x2Le |
		I8x16GtS |
		I8x16GtU |
		I16x8GtS |
		I16x8GtU |
		I32x4GtS |
		I32x4GtU |
		I64x2GtS |
		F32x4Gt |
		F64x2Gt |
		I8x16GeS |
		I8x16GeU |
		I16x8GeS |
		I16x8GeU |
		I32x4GeS |
		I32x4GeU |
		I64x2GeS |
		F32x4Ge |
		F64x2Ge |
		F32x4Neg |
		F64x2Neg |
		F32x4Min |
		F64x2Min |
		F32x4Max |
		F64x2Max |
		F32x4Add |
		F64x2Add |
		F32x4Sub |
		F64x2Sub |
		F32x4Mul |
		F64x2Mul |
		I8x16Swizzle |
		V128Andnot |
		I8x16NarrowI16x8S |
		I8x16NarrowI16x8U |
		F32x4Nearest |
		I8x16MinS |
		I8x16MinU |
		I8x16MaxS |
		I8x16MaxU |
		I8x16AvgrU |
		I16x8Q15mulrSatS |
		I16x8NarrowI32x4S |
		I16x8NarrowI32x4U |
		F64x2Nearest |
		I16x8MinS |
		I16x8MinU |
		I16x8MaxS |
		I16x8MaxU |
		I16x8AvgrU |
		I16x8ExtmulLowI8x16S |
		I16x8ExtmulHighI8x16S |
		I16x8ExtmulLowI8x16U |
		I16x8ExtmulHighI8x16U |
		I32x4MinS |
		I32x4MinU |
		I32x4MaxS |
		I32x4MaxU |
		I32x4DotI16x8S |
		I32x4ExtmulLowI16x8S |
		I32x4ExtmulHighI16x8S |
		I32x4ExtmulLowI16x8U |
		I32x4ExtmulHighI16x8U |
		I64x2ExtmulLowI32x4S |
		I64x2ExtmulHighI32x4S |
		I64x2ExtmulLowI32x4U |
		I64x2ExtmulHighI32x4U |
		F32x4Pmin |
		F32x4Pmax |
		F64x2Pmin |
		F64x2Pmax => { stack.pop_values(2)?; stack.push_values(1)?; }
		V128Bitselect => { stack.pop_values(3)?; stack.push_values(1)?; }
	}
	Ok(())
}
#[cfg(all(test, feature = "bulk", feature = "simd"))]
mod opcode_tests {
	use parity_wasm::elements;

	fn roundtrip(wat_src: &str) {
		let wasm = wat::parse_str(wat_src).expect("wat");
		let module: elements::Module = elements::deserialize_buffer(&wasm).expect("decode");
		let out = crate::inject_stack_limiter(module, 4096, &Default::default()).expect("limit");
		let bytes = elements::serialize(out).expect("encode");
		elements::deserialize_buffer::<elements::Module>(&bytes).expect("redecode");
	}

	#[test]
	fn bulk_memory_copy_passes_the_limiter() {
		roundtrip(
			r#"(module
				(memory 1)
				(func (export "c")
					(memory.copy (i32.const 0) (i32.const 0) (i32.const 4))))"#,
		);
	}

	#[test]
	fn simd_trunc_sat_f64x2_passes_the_limiter() {
		roundtrip(
			r#"(module
				(func (export "t") (param v128) (result v128)
					(i32x4.trunc_sat_f64x2_s_zero (local.get 0))))"#,
		);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use parity_wasm::elements;

	fn parse_wat(source: &str) -> elements::Module {
		elements::deserialize_buffer(&wat::parse_str(source).expect("Failed to wat2wasm"))
			.expect("Failed to deserialize the module")
	}

	#[test]
	fn simple_test() {
		let module = parse_wat(
			r#"
(module
	(func
		i32.const 1
			i32.const 2
				i32.const 3
				drop
			drop
		drop
	)
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 3 + ACTIVATION_FRAME_COST);
	}

	#[test]
	fn implicit_and_explicit_return() {
		let module = parse_wat(
			r#"
(module
	(func (result i32)
		i32.const 0
		return
	)
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 1 + ACTIVATION_FRAME_COST);
	}

	#[test]
	fn dont_count_in_unreachable() {
		let module = parse_wat(
			r#"
(module
  (memory 0)
  (func (result i32)
	unreachable
	memory.grow
  )
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, ACTIVATION_FRAME_COST);
	}

	#[test]
	fn yet_another_test() {
		let module = parse_wat(
			r#"
(module
  (memory 0)
  (func
	;; Push two values and then pop them.
	;; This will make max depth to be equal to 2.
	i32.const 0
	i32.const 1
	drop
	drop

	;; Code after `unreachable` shouldn't have an effect
	;; on the max depth.
	unreachable
	i32.const 0
	i32.const 1
	i32.const 2
  )
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 2 + ACTIVATION_FRAME_COST);
	}

	#[test]
	fn call_indirect() {
		let module = parse_wat(
			r#"
(module
	(table $ptr 1 1 funcref)
	(elem $ptr (i32.const 0) func 1)
	(func $main
		(call_indirect (i32.const 0))
		(call_indirect (i32.const 0))
		(call_indirect (i32.const 0))
	)
	(func $callee
		i64.const 42
		drop
	)
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 1 + ACTIVATION_FRAME_COST);
	}

	#[test]
	fn breaks() {
		let module = parse_wat(
			r#"
(module
	(func $main
		block (result i32)
			block (result i32)
				i32.const 99
				br 1
			end
		end
		drop
	)
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 1 + ACTIVATION_FRAME_COST);
	}

	#[test]
	fn if_else_works() {
		let module = parse_wat(
			r#"
(module
	(func $main
		i32.const 7
		i32.const 1
		if (result i32)
			i32.const 42
		else
			i32.const 99
		end
		i32.const 97
		drop
		drop
		drop
	)
)
"#,
		);

		let height = compute(0, &module).unwrap();
		assert_eq!(height, 3 + ACTIVATION_FRAME_COST);
	}
}
