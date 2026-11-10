use std::ops::IndexMut;

use super::{KeyString, Value};

impl Value {
    /// Create an iterator over the `Value`.
    ///
    /// For non-collection types, this returns a single-item iterator similar to
    /// `Option`'s iterator implementation.
    ///
    /// For collection types, it returns all elements in the collection.
    ///
    /// The resulting item is an [`IterItem`], which contains either a mutable
    /// `Value` for non-collection types, a (`&mut KeyString`, `&mut Value`) pair for
    /// object-type collections, or an immutable/mutable (`usize`, `&mut Value`)
    /// pair for array-type collections.
    ///
    /// ## Recursion
    ///
    /// If `recursion` is set to `true`, the iterator recurses into nested
    /// collection types.
    ///
    /// Recursion follows these rules:
    ///
    /// - When a collection type is found, that type is first returned as-is.
    ///   That is, if we're iterating over an object, and within that object is
    ///   a field "foo" containing an array, then we first return
    ///   `IterItem::KeyValue`, which contains the key "foo", and the array as
    ///   the value.
    /// - After returning the collection type, the iterator recurses into the
    ///   nested collection itself. Using the previous example, we now go into
    ///   the array, and start returning `IterItem::IndexValue` variants for the
    ///   elements within the array.
    /// - Any mutations done to the array before recursing into it are
    ///   preserved, meaning once recursion starts, the mutations done on the
    ///   object itself are preserved.
    pub fn into_iter(self, recursive: bool) -> ValueIter {
        let data = match self {
            Self::Object(object) => IterData::Object(object.into_iter().collect()),
            Self::Array(array) => IterData::Array(array),
            value => IterData::Value(value),
        };

        ValueIter::new(data, recursive)
    }
}

/// A lending iterator over a [`Value`].
pub struct ValueIter {
    data: IterData,
    recursive: bool,
    index: usize,
    recursive_iter: Option<Box<Self>>,
    must_prepare_recursion: bool,
    in_recursive: bool,
}

/// The item returned by [`ValueIter::next`].
#[derive(Debug, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)]
pub enum IterItem<'a> {
    /// A single primitive value.
    Value(&'a mut Value),

    /// A key/value combination.
    KeyValue(&'a mut KeyString, &'a mut Value),

    /// An index/value combination.
    IndexValue(usize, &'a mut Value),
}

/// The internal data representation used by the iterator.
///
/// Objects are stored as a `Vec<(KeyString, Value)>`, to allow mutating the object
/// keys during iteration.
enum IterData {
    Value(Value),
    Object(Vec<(KeyString, Value)>),
    Array(Vec<Value>),
}

impl ValueIter {
    /// Create a new iterator over the relevant [`IterData`].
    fn new(data: IterData, recursive: bool) -> Self {
        Self {
            data,
            recursive,
            index: 0,
            recursive_iter: None,
            must_prepare_recursion: false,
            in_recursive: false,
        }
    }
}

impl ValueIter {
    /// Return the next item, borrowed from the iterator itself.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<IterItem<'_>> {
        if self.advance() {
            self.current()
        } else {
            None
        }
    }

    /// Move to the next item, returning false when exhausted.
    fn advance(&mut self) -> bool {
        // If this returns true, it means on the last iteration cycle, we've
        // returned a collection value-type, and the caller asked to recurse
        // into that collection.
        //
        // We're going to prepare recursion by embedding a new recursive
        // iterator into `ValueIter`. This preparation is delayed to this new
        // cycle, to allow the caller to first mutate the collection we're going
        // to recurse into.
        if self.must_prepare_recursion {
            self.must_prepare_recursion = false;

            // Get the relevant collection `Value` type we want to iterate over.
            let value = match &mut self.data {
                IterData::Object(object) => object.get_mut(self.index - 1).map(|kv| &mut kv.1),
                IterData::Array(array) => array.get_mut(self.index - 1),
                IterData::Value(_) => unreachable!("cannot recurse into non-container type"),
            };
            let Some(value) = value else { return false };

            // Create a new `IterData` type we're going to embed recursively
            // into this iterator, to allow iterating over.
            let data = match value {
                Value::Object(object) => {
                    Some(IterData::Object(object.clone().into_iter().collect()))
                }
                Value::Array(array) => Some(IterData::Array(array.clone())),

                // It's possible the [`Value`] we're trying to iterate over is
                // a non-collection type. This happens if the caller changed the
                // value type of the collection to a non-collection type in the
                // last iteration cycle.
                //
                // That is, given this:
                //
                // ```
                // { "foo": [true] }
                // ```
                //
                // If they iterate over `(foo, [true])` and change `[true]` to
                // a non-collection type (e.g. `null`), then this branch gets
                // hit, and we abort the recursion.
                _ => None,
            };

            self.recursive_iter = data.map(|data| Box::new(Self::new(data, self.recursive)));
        }

        // If we have a recursive iterator stored, it means we're asked to
        // continue iterating that iterator until it is exhausted.
        self.in_recursive = false;
        if let Some(iter) = &mut self.recursive_iter {
            // Keep returning recursive items until the iterator is exhausted.
            if iter.advance() {
                self.in_recursive = true;
                return true;
            }

            // Now that we're done recursing, we need to update the relevant `Value`
            // collection type stored in this iterator, with the updated
            // inner elements that the recursive iterator might have
            // mutated.
            let value = match &mut self.data {
                IterData::Object(object) => &mut object.index_mut(self.index - 1).1,
                IterData::Array(array) => array.index_mut(self.index - 1),
                IterData::Value(_) => {
                    unreachable!("cannot recurse into non-container type")
                }
            };

            *value = (*self.recursive_iter.take().unwrap()).into();
        }

        // If we got here, we are either done with recursively iterating the
        // active value, or we haven't started recursively iterating yet (which
        // would be the case if we haven't encountered a collection `Value` type
        // yet).
        //
        // We only want to recurse deeper into nested collections, if requested
        // by the caller.
        let is_collection = match &self.data {
            IterData::Object(object) => match object.get(self.index) {
                Some((_, value)) => value.is_object() || value.is_array(),
                None => return false,
            },
            IterData::Array(array) => match array.get(self.index) {
                Some(value) => value.is_object() || value.is_array(),
                None => return false,
            },
            // We check if `self.index == 0` as a means to ensure we ever only
            // return this value once.
            IterData::Value(_) if self.index == 0 => false,
            IterData::Value(_) => return false,
        };

        self.must_prepare_recursion = is_collection && self.recursive;
        self.index += 1;

        true
    }

    /// Borrow the item that `advance` last moved to.
    fn current(&mut self) -> Option<IterItem<'_>> {
        if self.in_recursive {
            return self.recursive_iter.as_mut()?.current();
        }

        let index = self.index.checked_sub(1)?;

        match &mut self.data {
            IterData::Object(object) => {
                let (key, value) = object.get_mut(index)?;
                Some(IterItem::KeyValue(key, value))
            }
            IterData::Array(array) => Some(IterItem::IndexValue(index, array.get_mut(index)?)),
            IterData::Value(value) => Some(IterItem::Value(value)),
        }
    }
}

impl From<ValueIter> for Value {
    fn from(iter: ValueIter) -> Self {
        iter.data.into()
    }
}

impl From<IterData> for Value {
    fn from(iter: IterData) -> Self {
        match iter {
            IterData::Value(value) => value,
            IterData::Object(object) => Self::Object(object.into_iter().collect()),
            IterData::Array(array) => Self::Array(array),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use super::*;

    #[test]
    #[allow(clippy::too_many_lines)]
    fn test_non_recursive() {
        struct TestCase {
            value: Value,
            recursive: bool,
            items: Vec<Value>,
        }

        for (
            title,
            TestCase {
                value,
                recursive,
                items,
            },
        ) in HashMap::from([
            (
                "null",
                TestCase {
                    value: Value::Null,
                    recursive: false,
                    items: vec![Value::Null],
                },
            ),
            (
                "null recursive",
                TestCase {
                    value: Value::Null,
                    recursive: true,
                    items: vec![Value::Null],
                },
            ),
            (
                "bool",
                TestCase {
                    value: Value::Boolean(true),
                    recursive: false,
                    items: vec![Value::Boolean(true)],
                },
            ),
            (
                "object non-recursive",
                TestCase {
                    value: Value::Object(BTreeMap::from([("foo".into(), true.into())])),
                    recursive: false,
                    items: vec![true.into()],
                },
            ),
            (
                "object recursive",
                TestCase {
                    value: BTreeMap::from([(
                        "foo".into(),
                        BTreeMap::from([("foo".into(), true.into()), ("bar".into(), "baz".into())])
                            .into(),
                    )])
                    .into(),
                    recursive: true,
                    items: vec![
                        BTreeMap::from([("foo".into(), true.into()), ("bar".into(), "baz".into())])
                            .into(),
                        "baz".into(),
                        true.into(),
                    ],
                },
            ),
            (
                "object multi-recursive",
                TestCase {
                    value: BTreeMap::from([
                        (
                            "foo".into(),
                            BTreeMap::from([("bar".into(), Value::Null)]).into(),
                        ),
                        ("baz".into(), true.into()),
                    ])
                    .into(),
                    recursive: true,
                    items: vec![
                        true.into(),
                        BTreeMap::from([("bar".into(), Value::Null)]).into(),
                        Value::Null,
                    ],
                },
            ),
        ]) {
            let mut iter = value.into_iter(recursive);
            let mut got = vec![];
            while let Some(item) = iter.next() {
                got.push(match item {
                    IterItem::Value(value) => value.clone(),
                    IterItem::KeyValue(_key, value) => value.clone(),
                    IterItem::IndexValue(..) => todo!(),
                });
            }

            assert_eq!(got, items, "{title}");
        }
    }

    #[test]
    fn test_mutations() {
        let data: Value = BTreeMap::from([("foo".into(), vec![true].into())]).into();

        // Empty vec before recursing means recursion doesn't find any elements.
        let mut iter = data.clone().into_iter(true);
        let mut iterations = 0;
        let mut cleared = false;
        let mut values = vec![];

        while let Some(item) = iter.next() {
            iterations += 1;

            match item {
                IterItem::Value(value) => values.push(value.clone()),
                IterItem::KeyValue(key, value) => {
                    if let Value::Array(array) = value {
                        array.clear();
                        cleared = true;
                    }
                }
                IterItem::IndexValue(index, value) => values.push(value.clone()),
            }
        }

        assert_eq!(iterations, 1);
        assert_eq!(values, vec![]);
        assert!(cleared);

        // Change vec type to non-collection before recursion means recursion
        // doesn't happen.
        let mut changed = false;
        iter = data.clone().into_iter(true);
        iterations = 0;
        values = vec![];

        while let Some(item) = iter.next() {
            iterations += 1;

            match item {
                IterItem::Value(value) => values.push(value.clone()),
                IterItem::KeyValue(key, value) => {
                    if let value @ Value::Array(..) = value {
                        *value = Value::Null;
                        changed = true;
                    }
                }
                IterItem::IndexValue(index, value) => values.push(value.clone()),
            }
        }

        assert_eq!(iterations, 1);
        assert_eq!(values, vec![]);
        assert!(changed);

        // Change vec type to a different collection type before recursion means
        // recursion keeps working as expected
        iter = data.into_iter(true);
        changed = false;
        iterations = 0;
        values = vec![];

        while let Some(item) = iter.next() {
            iterations += 1;

            match item {
                IterItem::Value(value) => values.push(value.clone()),
                IterItem::KeyValue(key, value) => match value {
                    value @ Value::Array(..) => {
                        *value = BTreeMap::from([("bar".into(), true.into())]).into();
                        changed = true;
                    }
                    value => values.push(value.clone()),
                },
                IterItem::IndexValue(index, value) => values.push(value.clone()),
            }
        }

        assert_eq!(iterations, 2);
        assert_eq!(values, vec![true.into()]);
        assert!(changed);
    }

    #[test]
    fn test_recursive_mutation_written_back() {
        let data: Value = BTreeMap::from([("foo".into(), vec![Value::from(1)].into())]).into();

        let mut iter = data.into_iter(true);
        while let Some(item) = iter.next() {
            if let IterItem::IndexValue(_, value) = item {
                *value = Value::from(2);
            }
        }

        let expected: Value = BTreeMap::from([("foo".into(), vec![Value::from(2)].into())]).into();
        assert_eq!(Value::from(iter), expected);
    }
}
