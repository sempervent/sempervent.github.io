# Rust Unsafe Programming Best Practices

**Objective**: Master senior-level Rust unsafe programming patterns for production systems. When you need to interface with C code, when you want to optimize performance, when you need enterprise-grade unsafe patterns—these best practices become your weapon of choice.

## Core Principles

- **Minimize Unsafe**: Use unsafe only when necessary
- **Safe Abstractions**: Wrap unsafe code in safe interfaces
- **Documentation**: Document all unsafe invariants
- **Testing**: Thoroughly test unsafe code
- **Audit**: Regularly audit unsafe code

## Unsafe Patterns

### Basic Unsafe Operations

```rust
// rust/01-basic-unsafe.rs

/*
Basic unsafe patterns and best practices for Rust
*/

use std::ptr;
use std::mem;
use std::alloc::{alloc, dealloc, Layout};

/// Safe wrapper around unsafe memory operations.
pub struct SafeBuffer {
    ptr: *mut u8,
    len: usize,
    capacity: usize,
}

impl SafeBuffer {
    /// Creates a new buffer with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        let layout = Layout::from_size_align(capacity, 1).unwrap();
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            panic!("Failed to allocate memory");
        }
        
        Self {
            ptr,
            len: 0,
            capacity,
        }
    }
    
    /// Pushes a byte to the buffer.
    pub fn push(&mut self, byte: u8) -> Result<(), &'static str> {
        if self.len >= self.capacity {
            return Err("Buffer is full");
        }
        
        unsafe {
            ptr::write(self.ptr.add(self.len), byte);
        }
        
        self.len += 1;
        Ok(())
    }
    
    /// Gets a byte from the buffer.
    pub fn get(&self, index: usize) -> Option<u8> {
        if index >= self.len {
            return None;
        }
        
        unsafe {
            Some(ptr::read(self.ptr.add(index)))
        }
    }
    
    /// Gets a slice of the buffer.
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
    }
    
    /// Gets a mutable slice of the buffer.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.len)
        }
    }
}

impl Drop for SafeBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let layout = Layout::from_size_align(self.capacity, 1).unwrap();
            unsafe {
                dealloc(self.ptr, layout);
            }
        }
    }
}

/// Safe wrapper around raw pointers.
pub struct SafePointer<T> {
    ptr: *mut T,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> SafePointer<T> {
    /// Creates a new safe pointer from a raw pointer.
    pub fn new(ptr: *mut T) -> Self {
        Self {
            ptr,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Gets a reference to the value.
    pub fn get(&self) -> Option<&T> {
        if self.ptr.is_null() {
            None
        } else {
            unsafe {
                Some(&*self.ptr)
            }
        }
    }
    
    /// Gets a mutable reference to the value.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.ptr.is_null() {
            None
        } else {
            unsafe {
                Some(&mut *self.ptr)
            }
        }
    }
    
    /// Sets the value at the pointer.
    pub fn set(&mut self, value: T) {
        if !self.ptr.is_null() {
            unsafe {
                ptr::write(self.ptr, value);
            }
        }
    }
}

/// Safe wrapper around raw arrays.
pub struct SafeArray<T> {
    ptr: *mut T,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> SafeArray<T> {
    /// Creates a new safe array.
    pub fn new(len: usize) -> Self {
        let layout = Layout::array::<T>(len).unwrap();
        let ptr = unsafe { alloc(layout) as *mut T };
        
        if ptr.is_null() {
            panic!("Failed to allocate memory");
        }
        
        Self {
            ptr,
            len,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Gets an element by index.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            None
        } else {
            unsafe {
                Some(&*self.ptr.add(index))
            }
        }
    }
    
    /// Gets a mutable element by index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len {
            None
        } else {
            unsafe {
                Some(&mut *self.ptr.add(index))
            }
        }
    }
    
    /// Sets an element by index.
    pub fn set(&mut self, index: usize, value: T) -> Result<(), &'static str> {
        if index >= self.len {
            Err("Index out of bounds")
        } else {
            unsafe {
                ptr::write(self.ptr.add(index), value);
            }
            Ok(())
        }
    }
}

impl<T> Drop for SafeArray<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let layout = Layout::array::<T>(self.len).unwrap();
            unsafe {
                dealloc(self.ptr as *mut u8, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_buffer() {
        let mut buffer = SafeBuffer::new(10);
        buffer.push(1).unwrap();
        buffer.push(2).unwrap();
        buffer.push(3).unwrap();
        
        assert_eq!(buffer.get(0), Some(1));
        assert_eq!(buffer.get(1), Some(2));
        assert_eq!(buffer.get(2), Some(3));
        assert_eq!(buffer.get(3), None);
    }
    
    #[test]
    fn test_safe_pointer() {
        let mut value = 42;
        let mut pointer = SafePointer::new(&mut value as *mut i32);
        
        assert_eq!(pointer.get(), Some(&42));
        pointer.set(100);
        assert_eq!(pointer.get(), Some(&100));
    }
    
    #[test]
    fn test_safe_array() {
        let mut array = SafeArray::new(5);
        array.set(0, 1).unwrap();
        array.set(1, 2).unwrap();
        array.set(2, 3).unwrap();
        
        assert_eq!(array.get(0), Some(&1));
        assert_eq!(array.get(1), Some(&2));
        assert_eq!(array.get(2), Some(&3));
        assert_eq!(array.get(5), None);
    }
}
```

### FFI Patterns

```rust
// rust/02-ffi-patterns.rs

/*
FFI patterns and best practices for Rust
*/

use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_int};

/// Safe wrapper around C string operations.
pub struct SafeCString {
    ptr: *mut c_char,
}

impl SafeCString {
    /// Creates a new safe C string from a Rust string.
    pub fn new(s: &str) -> Result<Self, std::ffi::NulError> {
        let c_string = CString::new(s)?;
        let ptr = c_string.into_raw();
        Ok(Self { ptr })
    }
    
    /// Gets the C string as a Rust string.
    pub fn to_string(&self) -> Result<String, std::str::Utf8Error> {
        unsafe {
            let c_str = CStr::from_ptr(self.ptr);
            Ok(c_str.to_str()?.to_string())
        }
    }
    
    /// Gets the raw pointer.
    pub fn as_ptr(&self) -> *const c_char {
        self.ptr
    }
}

impl Drop for SafeCString {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = CString::from_raw(self.ptr);
            }
        }
    }
}

/// Safe wrapper around C function calls.
pub struct SafeCFunction;

impl SafeCFunction {
    /// Calls a C function with error handling.
    pub fn call_c_function(input: &str) -> Result<i32, String> {
        let c_string = CString::new(input)
            .map_err(|e| format!("Failed to create C string: {}", e))?;
        
        let result = unsafe {
            // Simulate C function call
            c_function_wrapper(c_string.as_ptr())
        };
        
        if result < 0 {
            Err("C function returned error".to_string())
        } else {
            Ok(result)
        }
    }
}

/// Safe wrapper around C struct operations.
pub struct SafeCStruct {
    data: Vec<u8>,
}

impl SafeCStruct {
    /// Creates a new safe C struct.
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }
    
    /// Gets a field from the struct.
    pub fn get_field(&self, offset: usize, size: usize) -> Option<&[u8]> {
        if offset + size <= self.data.len() {
            Some(&self.data[offset..offset + size])
        } else {
            None
        }
    }
    
    /// Sets a field in the struct.
    pub fn set_field(&mut self, offset: usize, data: &[u8]) -> Result<(), &'static str> {
        if offset + data.len() > self.data.len() {
            return Err("Field would exceed struct bounds");
        }
        
        self.data[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }
    
    /// Gets the raw pointer to the struct.
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }
}

// External C function declaration
extern "C" {
    fn c_function_wrapper(input: *const c_char) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_c_string() {
        let c_string = SafeCString::new("Hello, World!").unwrap();
        assert_eq!(c_string.to_string().unwrap(), "Hello, World!");
    }
    
    #[test]
    fn test_safe_c_struct() {
        let mut c_struct = SafeCStruct::new(100);
        let data = b"Hello";
        c_struct.set_field(0, data).unwrap();
        
        let field = c_struct.get_field(0, 5).unwrap();
        assert_eq!(field, data);
    }
}
```

### Memory Management

```rust
// rust/03-memory-management.rs

/*
Memory management patterns and best practices for Rust
*/

use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

/// Safe wrapper around raw memory allocation.
pub struct SafeMemory {
    ptr: *mut u8,
    size: usize,
    layout: Layout,
}

impl SafeMemory {
    /// Allocates memory with the specified size and alignment.
    pub fn new(size: usize, align: usize) -> Result<Self, &'static str> {
        let layout = Layout::from_size_align(size, align)
            .map_err(|_| "Invalid layout")?;
        
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err("Failed to allocate memory");
        }
        
        Ok(Self { ptr, size, layout })
    }
    
    /// Writes data to the memory.
    pub fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), &'static str> {
        if offset + data.len() > self.size {
            return Err("Write would exceed memory bounds");
        }
        
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len());
        }
        
        Ok(())
    }
    
    /// Reads data from the memory.
    pub fn read(&self, offset: usize, len: usize) -> Result<Vec<u8>, &'static str> {
        if offset + len > self.size {
            return Err("Read would exceed memory bounds");
        }
        
        let mut result = vec![0; len];
        unsafe {
            ptr::copy_nonoverlapping(self.ptr.add(offset), result.as_mut_ptr(), len);
        }
        
        Ok(result)
    }
    
    /// Gets the raw pointer.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    
    /// Gets the mutable raw pointer.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for SafeMemory {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                dealloc(self.ptr, self.layout);
            }
        }
    }
}

/// Safe wrapper around memory-mapped operations.
pub struct SafeMemoryMap {
    ptr: *mut u8,
    size: usize,
}

impl SafeMemoryMap {
    /// Creates a new memory map.
    pub fn new(size: usize) -> Result<Self, &'static str> {
        let layout = Layout::from_size_align(size, 1)
            .map_err(|_| "Invalid layout")?;
        
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err("Failed to allocate memory");
        }
        
        Ok(Self { ptr, size })
    }
    
    /// Maps data to the memory.
    pub fn map(&mut self, data: &[u8]) -> Result<(), &'static str> {
        if data.len() > self.size {
            return Err("Data too large for memory map");
        }
        
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len());
        }
        
        Ok(())
    }
    
    /// Unmaps data from the memory.
    pub fn unmap(&mut self) {
        unsafe {
            ptr::write_bytes(self.ptr, 0, self.size);
        }
    }
    
    /// Gets a slice of the mapped memory.
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.size)
        }
    }
}

impl Drop for SafeMemoryMap {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            let layout = Layout::from_size_align(self.size, 1).unwrap();
            unsafe {
                dealloc(self.ptr, layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_safe_memory() {
        let mut memory = SafeMemory::new(100, 1).unwrap();
        let data = b"Hello, World!";
        memory.write(0, data).unwrap();
        
        let read_data = memory.read(0, data.len()).unwrap();
        assert_eq!(read_data, data);
    }
    
    #[test]
    fn test_safe_memory_map() {
        let mut memory_map = SafeMemoryMap::new(100).unwrap();
        let data = b"Hello, World!";
        memory_map.map(data).unwrap();
        
        let slice = memory_map.as_slice();
        assert_eq!(&slice[..data.len()], data);
    }
}
```

## TL;DR Runbook

### Quick Start

```rust
// 1. Basic unsafe operations
unsafe {
    let ptr = std::ptr::null_mut::<i32>();
    let value = std::ptr::read(ptr);
}

// 2. Safe wrappers
struct SafeWrapper {
    ptr: *mut u8,
}

impl SafeWrapper {
    fn new() -> Self {
        let layout = std::alloc::Layout::new::<u8>();
        let ptr = unsafe { std::alloc::alloc(layout) };
        Self { ptr }
    }
}

// 3. FFI patterns
extern "C" {
    fn c_function(input: *const i8) -> i32;
}

// 4. Memory management
unsafe {
    let layout = std::alloc::Layout::new::<i32>();
    let ptr = std::alloc::alloc(layout);
    std::alloc::dealloc(ptr, layout);
}
```

### Essential Patterns

```rust
// Complete unsafe programming setup
pub fn setup_rust_unsafe() {
    // 1. Unsafe operations
    // 2. Safe wrappers
    // 3. FFI patterns
    // 4. Memory management
    // 5. Raw pointers
    // 6. Memory allocation
    // 7. C interop
    // 8. Performance optimization
    
    println!("Rust unsafe programming setup complete!");
}
```

---

*This guide provides the complete machinery for Rust unsafe programming. Each pattern includes implementation examples, safety strategies, and real-world usage patterns for enterprise unsafe code.*
