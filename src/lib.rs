use core::ops::Shl as _;
use core::ops::Shr as _;

/// Available instructions of this dummy interpreter.
///
/// # Note
///
/// We need to add quite a few instructions so that the
/// difference between threaded-code dispatch and switch-loop
/// dispatch becomes visible in our benchmarks.
#[derive(Debug, Copy, Clone)]
pub enum Instruction {
    // Constants
    Const {
        bits: Bits,
    },
    // Local Variables
    LocalGet {
        index: LocalIndex,
    },
    LocalSet {
        index: LocalIndex,
    },
    LocalTee {
        index: LocalIndex,
    },
    /// Branches using the given offset to the instruction pointer.
    Br {
        offset: BranchOffset,
    },
    /// Branches only if the top most value on the stack is equal to zero.
    BrEqz {
        offset: BranchOffset,
    },
    /// Branches only if the top most value on the stack is not equal to zero.
    BrNez {
        offset: BranchOffset,
    },
    /// Returns the top most value on the stack. Ends execution.
    Return,
    /// Drops the amount of values from the stack.
    Drop,
    /// Duplicates the top most value on the stack.
    Dup,
    // Math
    // i32
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32DivU,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32ShrS,
    I32ShrU,
    // i64
    I64Add,
    I64Sub,
    I64Mul,
    I64DivS,
    I64DivU,
    I64And,
    I64Or,
    I64Xor,
    I64Shl,
    I64ShrS,
    I64ShrU,
    // Comparison
    // i32
    I32Eq,
    I32Ne,
    I32LeS,
    I32LtS,
    I32GeS,
    I32GtS,
    I32LeU,
    I32LtU,
    I32GeU,
    I32GtU,
    // i64
    I64Eq,
    I64Ne,
    I64LeS,
    I64LtS,
    I64GeS,
    I64GtS,
    I64LeU,
    I64LtU,
    I64GeU,
    I64GtU,
}

/// A trap or error during instruction execution.
#[derive(Debug, Copy, Clone)]
pub enum Trap {
    DivisionByZero,
}

/// Some untyped bits.
#[derive(Debug, Copy, Clone)]
pub struct Bits(u64);

impl Default for Bits {
    fn default() -> Self {
        Self(0)
    }
}

impl From<i32> for Bits {
    fn from(value: i32) -> Self {
        Self(value as u64)
    }
}

impl From<u32> for Bits {
    fn from(value: u32) -> Self {
        Self(value as u64)
    }
}

impl From<i64> for Bits {
    fn from(value: i64) -> Self {
        Self(value as u64)
    }
}

impl From<u64> for Bits {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl From<Bits> for i32 {
    fn from(bits: Bits) -> Self {
        bits.0 as _
    }
}

impl From<Bits> for u32 {
    fn from(bits: Bits) -> Self {
        bits.0 as _
    }
}

impl From<Bits> for i64 {
    fn from(bits: Bits) -> Self {
        bits.0 as _
    }
}

impl From<Bits> for u64 {
    fn from(bits: Bits) -> Self {
        bits.0
    }
}

/// The index of a local variable.
#[derive(Debug, Copy, Clone)]
pub struct LocalIndex(usize);

/// The branching offset.
#[derive(Debug, Copy, Clone)]
pub struct BranchOffset(isize);

/// The value stack.
///
/// # Note
///
/// For performance reasons we do not use a `Vec` as the underlying data structure.
#[derive(Debug)]
pub struct Stack {
    /// The values of the [`Stack`].
    values: Box<[Bits]>,
    /// The current stack pointer.
    sp: usize,
}

impl Stack {
    /// Creates a new value stack with the given size.
    pub fn new(len: usize) -> Self {
        let values = Box::from(vec![Bits::default(); len]);
        Self { values, sp: 0 }
    }

    /// Pushes a local variable to the [`Stack`].
    pub fn push_local(&mut self, value: Bits) -> LocalIndex {
        self.values[self.sp] = value;
        let index = LocalIndex(self.sp);
        self.sp += 1;
        index
    }

    /// Returns the value of the given local variable.
    pub fn local_get(&self, index: LocalIndex) -> Bits {
        self.values[index.0]
    }

    /// Sets the value of the given local variable to the given `bits`.
    pub fn local_set(&mut self, index: LocalIndex, new_value: Bits) {
        self.values[index.0] = new_value;
    }

    /// Pops the last value from the [`Stack`].
    ///
    /// # Panics
    ///
    /// If the [`Stack`] is empty.
    pub fn pop(&mut self) -> Bits {
        self.sp -= 1;
        self.values[self.sp]
    }

    /// Peeks the last value of the [`Stack`].
    pub fn peek(&self) -> Bits {
        self.values[self.sp - 1]
    }

    /// Peeks a mutable reference to the last value of the [`Stack`].
    pub fn peek_mut(&mut self) -> &mut Bits {
        &mut self.values[self.sp - 1]
    }

    /// Pushes the given `bits` to the [`Stack`].
    ///
    /// # Panics
    ///
    /// If the [`Stack`] is already full.
    pub fn push(&mut self, bits: Bits) {
        self.values[self.sp] = bits;
        self.sp += 1;
    }
}

/// Executes a list of instructions.
#[derive(Debug)]
pub struct Executor {
    /// The list of instructions to execute.
    instrs: Vec<Instruction>,
    /// The instruction pointer.
    ip: usize,
    /// The value stack.
    stack: Stack,
}

impl Default for Executor {
    fn default() -> Self {
        Self {
            instrs: Vec::new(),
            ip: 0,
            stack: Stack::new(100),
        }
    }
}

impl Executor {
    /// Pushes a local variable with the given `value` to the [`Executor`].
    pub fn push_local(&mut self, value: Bits) -> LocalIndex {
        self.stack.push_local(value)
    }

    /// Pushes a bunch of instructions and returns a [`Func`] that refers to them.
    pub fn push_instrs<I>(&mut self, instrs: I) -> Func
    where
        I: IntoIterator<Item = Instruction>,
    {
        let index = self.instrs.len();
        self.instrs.extend(instrs);
        Func(index)
    }

    /// Executes the instructions of the [`Executor`] and returns the result.
    pub fn execute(&mut self) -> Result<Bits, Trap> {
        use Instruction as Instr;
        loop {
            let instr = self.instrs[self.ip];
            match instr {
                Instr::Const { bits } => self.execute_const(bits),
                Instr::LocalGet { index } => self.execute_local_get(index),
                Instr::LocalSet { index } => self.execute_local_set(index),
                Instr::LocalTee { index } => self.execute_local_tee(index),
                Instr::Br { offset } => self.execute_br(offset),
                Instr::BrEqz { offset } => self.execute_br_eqz(offset),
                Instr::BrNez { offset } => self.execute_br_nez(offset),
                Instr::Return => return Ok(self.execute_return()),
                Instr::Drop => self.execute_drop(),
                Instr::Dup => self.execute_dup(),
                Instr::I32Add => self.execute_i32_add(),
                Instr::I32Sub => self.execute_i32_sub(),
                Instr::I32Mul => self.execute_i32_mul(),
                Instr::I32DivS => self.execute_i32_div_s()?,
                Instr::I32DivU => self.execute_i32_div_u()?,
                Instr::I32And => self.execute_i32_and(),
                Instr::I32Or => self.execute_i32_or(),
                Instr::I32Xor => self.execute_i32_xor(),
                Instr::I32Shl => self.execute_i32_shl(),
                Instr::I32ShrS => self.execute_i32_shr_s(),
                Instr::I32ShrU => self.execute_i32_shr_u(),
                Instr::I64Add => self.execute_i64_add(),
                Instr::I64Sub => self.execute_i64_sub(),
                Instr::I64Mul => self.execute_i64_mul(),
                Instr::I64DivS => self.execute_i64_div_s()?,
                Instr::I64DivU => self.execute_i64_div_u()?,
                Instr::I64And => self.execute_i64_and(),
                Instr::I64Or => self.execute_i64_or(),
                Instr::I64Xor => self.execute_i64_xor(),
                Instr::I64Shl => self.execute_i64_shl(),
                Instr::I64ShrS => self.execute_i64_shr_s(),
                Instr::I64ShrU => self.execute_i64_shr_u(),
                Instr::I32Eq => self.execute_i32_eq(),
                Instr::I32Ne => self.execute_i32_ne(),
                Instr::I32LeS => self.execute_i32_lt_s(),
                Instr::I32LtS => self.execute_i32_le_s(),
                Instr::I32GeS => self.execute_i32_gt_s(),
                Instr::I32GtS => self.execute_i32_ge_s(),
                Instr::I32LeU => self.execute_i32_lt_u(),
                Instr::I32LtU => self.execute_i32_le_u(),
                Instr::I32GeU => self.execute_i32_gt_u(),
                Instr::I32GtU => self.execute_i32_ge_u(),
                Instr::I64Eq => self.execute_i64_eq(),
                Instr::I64Ne => self.execute_i64_ne(),
                Instr::I64LeS => self.execute_i64_lt_s(),
                Instr::I64LtS => self.execute_i64_le_s(),
                Instr::I64GeS => self.execute_i64_gt_s(),
                Instr::I64GtS => self.execute_i64_ge_s(),
                Instr::I64LeU => self.execute_i64_lt_u(),
                Instr::I64LtU => self.execute_i64_le_u(),
                Instr::I64GeU => self.execute_i64_gt_u(),
                Instr::I64GtU => self.execute_i64_ge_u(),
            }
        }
    }

    fn next_instr(&mut self) {
        self.ip += 1;
    }

    fn branch_to_instr(&mut self, offset: BranchOffset) {
        self.ip = (self.ip as isize + offset.0) as usize;
    }

    fn execute_const(&mut self, value: Bits) {
        self.stack.push(value);
        self.next_instr();
    }

    fn execute_local_get(&mut self, index: LocalIndex) {
        let value = self.stack.local_get(index);
        self.stack.push(value);
        self.next_instr();
    }

    fn execute_local_set(&mut self, index: LocalIndex) {
        let value = self.stack.pop();
        self.stack.local_set(index, value);
        self.next_instr();
    }

    fn execute_local_tee(&mut self, index: LocalIndex) {
        let value = self.stack.peek();
        self.stack.local_set(index, value);
        self.next_instr();
    }

    fn execute_br(&mut self, offset: BranchOffset) {
        self.branch_to_instr(offset);
    }

    fn execute_br_eqz(&mut self, offset: BranchOffset) {
        let condition = self.stack.pop();
        if i32::from(condition) == 0 {
            self.branch_to_instr(offset);
        } else {
            self.next_instr();
        }
    }

    fn execute_br_nez(&mut self, offset: BranchOffset) {
        let condition = self.stack.pop();
        if i32::from(condition) != 0 {
            self.branch_to_instr(offset);
        } else {
            self.next_instr();
        }
    }

    fn execute_return(&mut self) -> Bits {
        self.stack.pop()
    }

    fn execute_drop(&mut self) {
        self.stack.pop();
        self.next_instr();
    }

    fn execute_dup(&mut self) {
        self.stack.push(self.stack.peek());
        self.next_instr();
    }

    fn execute_binop(&mut self, f: fn(Bits, Bits) -> Bits) {
        let rhs = self.stack.pop();
        let lhs = self.stack.pop();
        let result = f(lhs, rhs);
        self.stack.push(result);
        self.next_instr();
    }

    fn execute_fallible_binop(
        &mut self,
        f: fn(Bits, Bits) -> Result<Bits, Trap>,
    ) -> Result<(), Trap> {
        let rhs = self.stack.pop();
        let lhs = self.stack.pop();
        let result = f(lhs, rhs)?;
        self.stack.push(result);
        self.next_instr();
        Ok(())
    }

    fn execute_i32_add(&mut self) {
        self.execute_binop(Bits::i32_add)
    }

    fn execute_i32_sub(&mut self) {
        self.execute_binop(Bits::i32_sub)
    }

    fn execute_i32_mul(&mut self) {
        self.execute_binop(Bits::i32_mul)
    }

    fn execute_i32_div_s(&mut self) -> Result<(), Trap> {
        self.execute_fallible_binop(Bits::i32_div_s)
    }

    fn execute_i32_div_u(&mut self) -> Result<(), Trap> {
        self.execute_fallible_binop(Bits::i32_div_u)
    }

    fn execute_i32_and(&mut self) {
        self.execute_binop(Bits::i32_and)
    }

    fn execute_i32_or(&mut self) {
        self.execute_binop(Bits::i32_or)
    }

    fn execute_i32_xor(&mut self) {
        self.execute_binop(Bits::i32_xor)
    }

    fn execute_i32_shl(&mut self) {
        self.execute_binop(Bits::i32_shl)
    }

    fn execute_i32_shr_s(&mut self) {
        self.execute_binop(Bits::i32_shr_s)
    }

    fn execute_i32_shr_u(&mut self) {
        self.execute_binop(Bits::i32_shr_u)
    }

    fn execute_i64_add(&mut self) {
        self.execute_binop(Bits::i64_add)
    }

    fn execute_i64_sub(&mut self) {
        self.execute_binop(Bits::i64_sub)
    }

    fn execute_i64_mul(&mut self) {
        self.execute_binop(Bits::i64_mul)
    }

    fn execute_i64_div_s(&mut self) -> Result<(), Trap> {
        self.execute_fallible_binop(Bits::i64_div_s)
    }

    fn execute_i64_div_u(&mut self) -> Result<(), Trap> {
        self.execute_fallible_binop(Bits::i64_div_u)
    }

    fn execute_i64_and(&mut self) {
        self.execute_binop(Bits::i64_and)
    }

    fn execute_i64_or(&mut self) {
        self.execute_binop(Bits::i64_or)
    }

    fn execute_i64_xor(&mut self) {
        self.execute_binop(Bits::i64_xor)
    }

    fn execute_i64_shl(&mut self) {
        self.execute_binop(Bits::i64_shl)
    }

    fn execute_i64_shr_s(&mut self) {
        self.execute_binop(Bits::i64_shr_s)
    }

    fn execute_i64_shr_u(&mut self) {
        self.execute_binop(Bits::i64_shr_u)
    }

    fn execute_i32_eq(&mut self) {
        self.execute_binop(Bits::i32_eq)
    }

    fn execute_i32_ne(&mut self) {
        self.execute_binop(Bits::i32_ne)
    }

    fn execute_i32_lt_s(&mut self) {
        self.execute_binop(Bits::i32_lt_s)
    }

    fn execute_i32_le_s(&mut self) {
        self.execute_binop(Bits::i32_le_s)
    }

    fn execute_i32_gt_s(&mut self) {
        self.execute_binop(Bits::i32_gt_s)
    }

    fn execute_i32_ge_s(&mut self) {
        self.execute_binop(Bits::i32_ge_s)
    }

    fn execute_i32_lt_u(&mut self) {
        self.execute_binop(Bits::i32_lt_u)
    }

    fn execute_i32_le_u(&mut self) {
        self.execute_binop(Bits::i32_le_u)
    }

    fn execute_i32_gt_u(&mut self) {
        self.execute_binop(Bits::i32_gt_u)
    }

    fn execute_i32_ge_u(&mut self) {
        self.execute_binop(Bits::i32_ge_u)
    }

    fn execute_i64_eq(&mut self) {
        self.execute_binop(Bits::i64_eq)
    }

    fn execute_i64_ne(&mut self) {
        self.execute_binop(Bits::i64_ne)
    }

    fn execute_i64_lt_s(&mut self) {
        self.execute_binop(Bits::i64_lt_s)
    }

    fn execute_i64_le_s(&mut self) {
        self.execute_binop(Bits::i64_le_s)
    }

    fn execute_i64_gt_s(&mut self) {
        self.execute_binop(Bits::i64_gt_s)
    }

    fn execute_i64_ge_s(&mut self) {
        self.execute_binop(Bits::i64_ge_s)
    }

    fn execute_i64_lt_u(&mut self) {
        self.execute_binop(Bits::i64_lt_u)
    }

    fn execute_i64_le_u(&mut self) {
        self.execute_binop(Bits::i64_le_u)
    }

    fn execute_i64_gt_u(&mut self) {
        self.execute_binop(Bits::i64_gt_u)
    }

    fn execute_i64_ge_u(&mut self) {
        self.execute_binop(Bits::i64_ge_u)
    }
}

impl Bits {
    fn i32_op(self, rhs: Self, f: fn(i32, i32) -> i32) -> Bits {
        Self::from(f(i32::from(self), i32::from(rhs)))
    }

    fn u32_op(self, rhs: Self, f: fn(u32, u32) -> u32) -> Bits {
        Self::from(f(u32::from(self), u32::from(rhs)))
    }

    pub fn i32_add(self, rhs: Self) -> Self {
        self.i32_op(rhs, i32::wrapping_add)
    }

    pub fn i32_sub(self, rhs: Self) -> Self {
        self.i32_op(rhs, i32::wrapping_sub)
    }

    pub fn i32_mul(self, rhs: Self) -> Self {
        self.i32_op(rhs, i32::wrapping_mul)
    }

    pub fn i32_div_s(self, rhs: Self) -> Result<Self, Trap> {
        let rhs = i32::from(rhs);
        if rhs == 0 {
            return Err(Trap::DivisionByZero);
        }
        let lhs = i32::from(self);
        Ok(Self::from(lhs / rhs))
    }

    pub fn i32_div_u(self, rhs: Self) -> Result<Self, Trap> {
        let rhs = u32::from(rhs);
        if rhs == 0 {
            return Err(Trap::DivisionByZero);
        }
        let lhs = u32::from(self);
        Ok(Self::from(lhs / rhs))
    }

    pub fn i32_and(self, rhs: Self) -> Self {
        self.i32_op(rhs, <i32 as core::ops::BitAnd>::bitand)
    }

    pub fn i32_or(self, rhs: Self) -> Self {
        self.i32_op(rhs, <i32 as core::ops::BitOr>::bitor)
    }

    pub fn i32_xor(self, rhs: Self) -> Self {
        self.i32_op(rhs, <i32 as core::ops::BitXor>::bitxor)
    }

    pub fn i32_shl(self, rhs: Self) -> Self {
        self.i32_op(rhs, |lhs: i32, rhs: i32| lhs.shl(rhs & 0x1F))
    }

    pub fn i32_shr_s(self, rhs: Self) -> Self {
        self.i32_op(rhs, |lhs: i32, rhs: i32| lhs.shr(rhs & 0x1F))
    }

    pub fn i32_shr_u(self, rhs: Self) -> Self {
        self.u32_op(rhs, |lhs: u32, rhs: u32| lhs.shr(rhs & 0x1F))
    }

    fn i64_op(self, rhs: Self, f: fn(i64, i64) -> i64) -> Bits {
        Self::from(f(i64::from(self), i64::from(rhs)))
    }

    fn u64_op(self, rhs: Self, f: fn(u64, u64) -> u64) -> Bits {
        Self::from(f(u64::from(self), u64::from(rhs)))
    }

    pub fn i64_add(self, rhs: Self) -> Self {
        self.i64_op(rhs, i64::wrapping_add)
    }

    pub fn i64_sub(self, rhs: Self) -> Self {
        self.i64_op(rhs, i64::wrapping_sub)
    }

    pub fn i64_mul(self, rhs: Self) -> Self {
        self.i64_op(rhs, i64::wrapping_mul)
    }

    pub fn i64_div_s(self, rhs: Self) -> Result<Self, Trap> {
        let rhs = i64::from(rhs);
        if rhs == 0 {
            return Err(Trap::DivisionByZero);
        }
        let lhs = i64::from(self);
        Ok(Self::from(lhs / rhs))
    }

    pub fn i64_div_u(self, rhs: Self) -> Result<Self, Trap> {
        let rhs = u64::from(rhs);
        if rhs == 0 {
            return Err(Trap::DivisionByZero);
        }
        let lhs = u64::from(self);
        Ok(Self::from(lhs / rhs))
    }

    pub fn i64_and(self, rhs: Self) -> Self {
        self.i64_op(rhs, <i64 as core::ops::BitAnd>::bitand)
    }

    pub fn i64_or(self, rhs: Self) -> Self {
        self.i64_op(rhs, <i64 as core::ops::BitOr>::bitor)
    }

    pub fn i64_xor(self, rhs: Self) -> Self {
        self.i64_op(rhs, <i64 as core::ops::BitXor>::bitxor)
    }

    pub fn i64_shl(self, rhs: Self) -> Self {
        self.i64_op(rhs, |lhs: i64, rhs: i64| lhs.shl(rhs & 0x1F))
    }

    pub fn i64_shr_s(self, rhs: Self) -> Self {
        self.i64_op(rhs, |lhs: i64, rhs: i64| lhs.shr(rhs & 0x1F))
    }

    pub fn i64_shr_u(self, rhs: Self) -> Self {
        self.u64_op(rhs, |lhs: u64, rhs: u64| lhs.shr(rhs & 0x1F))
    }

    pub fn i32_eq(self, rhs: Self) -> Self {
        self.i32_op(rhs, |lhs, rhs| (lhs == rhs) as i32)
    }

    pub fn i32_ne(self, rhs: Self) -> Self {
        self.i32_op(rhs, |lhs, rhs| (lhs != rhs) as i32)
    }

    pub fn i32_lt_s(self, rhs: Self) -> Self {
        self.i32_op(rhs, |lhs, rhs| (lhs < rhs) as i32)
    }

    pub fn i32_le_s(self, rhs: Self) -> Self {
        self.i32_op(rhs, |lhs, rhs| (lhs < rhs) as i32)
    }

    pub fn i32_gt_s(self, rhs: Self) -> Self {
        self.i32_op(rhs, |lhs, rhs| (lhs < rhs) as i32)
    }

    pub fn i32_ge_s(self, rhs: Self) -> Self {
        self.i32_op(rhs, |lhs, rhs| (lhs < rhs) as i32)
    }

    pub fn i32_lt_u(self, rhs: Self) -> Self {
        self.u32_op(rhs, |lhs, rhs| (lhs < rhs) as u32)
    }

    pub fn i32_le_u(self, rhs: Self) -> Self {
        self.u32_op(rhs, |lhs, rhs| (lhs < rhs) as u32)
    }

    pub fn i32_gt_u(self, rhs: Self) -> Self {
        self.u32_op(rhs, |lhs, rhs| (lhs < rhs) as u32)
    }

    pub fn i32_ge_u(self, rhs: Self) -> Self {
        self.u32_op(rhs, |lhs, rhs| (lhs < rhs) as u32)
    }

    pub fn i64_eq(self, rhs: Self) -> Self {
        self.i64_op(rhs, |lhs, rhs| (lhs == rhs) as i64)
    }

    pub fn i64_ne(self, rhs: Self) -> Self {
        self.i64_op(rhs, |lhs, rhs| (lhs != rhs) as i64)
    }

    pub fn i64_lt_s(self, rhs: Self) -> Self {
        self.i64_op(rhs, |lhs, rhs| (lhs < rhs) as i64)
    }

    pub fn i64_le_s(self, rhs: Self) -> Self {
        self.i64_op(rhs, |lhs, rhs| (lhs < rhs) as i64)
    }

    pub fn i64_gt_s(self, rhs: Self) -> Self {
        self.i64_op(rhs, |lhs, rhs| (lhs < rhs) as i64)
    }

    pub fn i64_ge_s(self, rhs: Self) -> Self {
        self.i64_op(rhs, |lhs, rhs| (lhs < rhs) as i64)
    }

    pub fn i64_lt_u(self, rhs: Self) -> Self {
        self.u64_op(rhs, |lhs, rhs| (lhs < rhs) as u64)
    }

    pub fn i64_le_u(self, rhs: Self) -> Self {
        self.u64_op(rhs, |lhs, rhs| (lhs < rhs) as u64)
    }

    pub fn i64_gt_u(self, rhs: Self) -> Self {
        self.u64_op(rhs, |lhs, rhs| (lhs < rhs) as u64)
    }

    pub fn i64_ge_u(self, rhs: Self) -> Self {
        self.u64_op(rhs, |lhs, rhs| (lhs < rhs) as u64)
    }
}

/// Refers to a bunch of instructions in the [`Executor`].
#[derive(Debug, Copy, Clone)]
pub struct Func(usize);

#[cfg(test)]
mod tests {
    use super::*;

    fn branch_offset(from: isize, to: isize) -> BranchOffset {
        BranchOffset(to as isize - from as isize)
    }

    #[test]
    fn benchmark() {
        let limit = 10_000_000;
        let mut ctx = Executor::default();
        let l0 = ctx.push_local(Bits(0));
        let l1 = ctx.push_local(Bits(0));
        let l2 = ctx.push_local(Bits(0));
        let instrs = [
            /* -2 */ Instruction::Const { bits: Bits(0) },
            /* -1 */ Instruction::Drop,
            /*  0 */ Instruction::LocalGet { index: l0 },
            /*  1 */ Instruction::LocalGet { index: l1 },
            /*  2 */ Instruction::I32Add, // l0 + l1
            /*  3 */ Instruction::Dup,
            /*  4 */ Instruction::I32Mul, // (l0 + l1) * (l0 + l1)
            /*  5 */ Instruction::Const { bits: Bits(2) },
            /*  6 */ Instruction::I32Shl, // >> 2 := divide by 2
            /*  7 */ Instruction::Drop,
            /*  8 */ Instruction::LocalGet { index: l2 },
            /*  9 */ Instruction::Const { bits: Bits(1) },
            /* 10 */ Instruction::I32Add,
            /* 11 */ Instruction::LocalTee { index: l2 },
            /* 12 */ Instruction::Const { bits: Bits(limit) },
            /* 13 */ Instruction::I32Eq,
            /* 14 */
            Instruction::BrEqz {
                offset: branch_offset(14, -2),
            },
            /* 15 */ Instruction::Return,
        ];
        ctx.push_instrs(instrs);
        let before = std::time::SystemTime::now();
        let result = ctx.execute().unwrap();
        let duration = before.elapsed().unwrap();
        println!("time: {duration:?}");
        println!("result = {:?}", result);
    }
}
