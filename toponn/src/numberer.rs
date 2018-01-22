use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;

use serde::{Deserialize, Serialize};

/// Numberer for categorical values, such as features or class labels.
#[derive(Eq, PartialEq, Serialize, Deserialize)]
pub struct Numberer<T>
where
    T: Eq + Hash + Serialize + Deserialize,
{
    values: Vec<T>,
    numbers: HashMap<T, usize>,
    start_at: usize,
}

impl<T> Numberer<T>
where
    T: Clone + Eq + Hash + Serialize + Deserialize,
{
    pub fn new(start_at: usize) -> Self {
        Numberer {
            values: Vec::new(),
            numbers: HashMap::new(),
            start_at: start_at,
        }
    }

    /// Add an value. If the value has already been encountered before,
    /// the corresponding number is returned.
    pub fn add(&mut self, value: T) -> usize {
        match self.numbers.entry(value.clone()) {
            Entry::Occupied(e) => e.get().clone(),
            Entry::Vacant(e) => {
                let number = self.values.len() + self.start_at;
                self.values.push(value);
                e.insert(number);
                number
            }
        }
    }

    /// Return the number for a value.
    pub fn number(&self, item: &T) -> Option<usize> {
        self.numbers.get(item).cloned()
    }

    /// Return the value for a number.
    pub fn value(&self, number: usize) -> Option<&T> {
        self.values.get(number - self.start_at)
    }
}
