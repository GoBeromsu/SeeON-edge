//! ONNX input dims (`worker/runtime/flow/onnx_shape.py`) without
//! onnxruntime. A minimal protobuf reader walks the one path Python reads:
//! `ModelProto.graph` (7), `GraphProto.input` (11, the first entry),
//! `ValueInfoProto.type` (2), `TypeProto.tensor_type` (1),
//! `TypeProto.Tensor.shape` (2), `TensorShapeProto.dim` (1), and each
//! `Dimension` as `dim_value` (1) or `dim_param` (2).

use std::fs;
use std::path::Path;

const MODEL_GRAPH: u64 = 7;
const GRAPH_INPUT: u64 = 11;
const VALUE_INFO_TYPE: u64 = 2;
const TYPE_TENSOR: u64 = 1;
const TENSOR_SHAPE: u64 = 2;
const SHAPE_DIM: u64 = 1;
const DIM_VALUE: u64 = 1;
const DIM_PARAM: u64 = 2;

/// The longest varint protobuf accepts.
const VARINT_BYTES: u32 = 10;

/// One input dimension as onnxruntime reports it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Dim {
    /// `dim_value`: Python sees an `int`.
    Value(i64),
    /// `dim_param`: Python sees the symbolic name.
    Param(String),
    /// Neither is set: Python sees `None`.
    Unknown,
}

/// Which `input_dims` refusal stopped the read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OnnxShapeKind {
    /// `cannot load ONNX input shape from {path}: {error}`, and the
    /// onnxruntime load errors that escape it unwrapped (`InvalidProtobuf`,
    /// `NoSuchFile`): the file is not a readable `ModelProto` with a graph.
    Unloadable,
    /// `ONNX artifact has no inputs: {path}`.
    NoInputs,
}

/// A refusal and the ONNX path it names.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OnnxShapeError {
    pub kind: OnnxShapeKind,
    pub subject: String,
}

/// One decoded field value. Fixed-width payloads are skipped.
enum Value<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed,
}

/// A cursor over one message's wire bytes. Every read is bounds-checked.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    /// A base-128 varint of at most ten bytes; bits past 64 are dropped.
    fn varint(&mut self) -> Option<u64> {
        let mut value = 0_u64;
        for index in 0..VARINT_BYTES {
            let (&byte, rest) = self.bytes.split_first()?;
            self.bytes = rest;
            value |= u64::from(byte & 0x7f).checked_shl(7 * index).unwrap_or(0);
            if byte & 0x80 == 0 {
                return Some(value);
            }
        }
        None
    }

    fn take(&mut self, len: usize) -> Option<&'a [u8]> {
        if len > self.bytes.len() {
            return None;
        }
        let (head, rest) = self.bytes.split_at(len);
        self.bytes = rest;
        Some(head)
    }

    /// One tag and its value. Field 0, a tag past 32 bits, groups and the
    /// reserved wire types 6 and 7 are malformed.
    fn field(&mut self) -> Option<(u64, Value<'a>)> {
        let tag = self.varint()?;
        let number = tag >> 3;
        if number == 0 || tag > u64::from(u32::MAX) {
            return None;
        }
        let value = match tag & 7 {
            0 => Value::Varint(self.varint()?),
            1 => self.take(8).map(|_| Value::Fixed)?,
            2 => {
                let len = usize::try_from(self.varint()?).ok()?;
                Value::Bytes(self.take(len)?)
            }
            5 => self.take(4).map(|_| Value::Fixed)?,
            _ => return None,
        };
        Some((number, value))
    }
}

/// Every field of one message, or `None` when its wire format is broken.
fn fields(message: &[u8]) -> Option<Vec<(u64, Value<'_>)>> {
    let mut reader = Reader { bytes: message };
    let mut found = Vec::new();
    while !reader.bytes.is_empty() {
        found.push(reader.field()?);
    }
    Some(found)
}

/// The length-delimited occurrences of field `number`, in wire order. A
/// known field with another wire type is an unknown field to protobuf.
fn messages<'a>(found: &[(u64, Value<'a>)], number: u64) -> Vec<&'a [u8]> {
    found
        .iter()
        .filter_map(|(field, value)| match value {
            Value::Bytes(bytes) if *field == number => Some(*bytes),
            _ => None,
        })
        .collect()
}

/// A singular embedded message: the last occurrence wins.
fn nested(message: &[u8], number: u64) -> Option<Option<&[u8]>> {
    fields(message).map(|found| messages(&found, number).last().copied())
}

/// One `Dimension`: the last of `dim_value` and `dim_param` wins (a oneof).
fn dim(message: &[u8]) -> Option<Dim> {
    let mut read = Dim::Unknown;
    for (number, value) in fields(message)? {
        read = match (number, value) {
            (DIM_VALUE, Value::Varint(raw)) => Dim::Value(raw.cast_signed()),
            (DIM_PARAM, Value::Bytes(text)) => Dim::Param(String::from_utf8(text.to_vec()).ok()?),
            _ => read,
        };
    }
    Some(read)
}

/// The dims of the first input's tensor shape; empty when the type or the
/// shape is not declared, as onnxruntime reports it. `None` is malformed.
fn first_input_dims(input: &[u8]) -> Option<Vec<Dim>> {
    let Some(kind) = nested(input, VALUE_INFO_TYPE)? else {
        return Some(Vec::new());
    };
    let Some(tensor) = nested(kind, TYPE_TENSOR)? else {
        return Some(Vec::new());
    };
    let Some(shape) = nested(tensor, TENSOR_SHAPE)? else {
        return Some(Vec::new());
    };
    let found = fields(shape)?;
    messages(&found, SHAPE_DIM).into_iter().map(dim).collect()
}

/// `input_dims(onnx_path)`: the first graph input's declared dims.
pub fn input_dims(path: &Path) -> Result<Vec<Dim>, OnnxShapeError> {
    let refuse = |kind| OnnxShapeError {
        kind,
        subject: path.to_string_lossy().into_owned(),
    };
    let unloadable = || refuse(OnnxShapeKind::Unloadable);
    let bytes = fs::read(path).map_err(|_| unloadable())?;
    let graph = nested(&bytes, MODEL_GRAPH)
        .flatten()
        .ok_or_else(unloadable)?;
    let found = fields(graph).ok_or_else(unloadable)?;
    let Some(input) = messages(&found, GRAPH_INPUT).first().copied() else {
        return Err(refuse(OnnxShapeKind::NoInputs));
    };
    first_input_dims(input).ok_or_else(unloadable)
}

/// `batch_axis_is_dynamic(dims)`: `not isinstance(dims[0], int)`. `None` is
/// the `IndexError` Python raises for an input with no dims.
pub fn batch_axis_is_dynamic(dims: &[Dim]) -> Option<bool> {
    dims.first().map(|first| !matches!(first, Dim::Value(_)))
}
