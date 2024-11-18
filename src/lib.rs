use std::sync::Arc;
use pyo3::prelude::*;
use numpy::PyReadonlyArrayDyn;
use arrow::{
    array::{Array, ArrayData, make_array, AsArray, UInt64Array},
    pyarrow::PyArrowType,
};


fn bisect_rust_imp<T>(list_input: &[T], value: &T, low: usize, high: Option<usize>) -> usize
where for<'a> &'a T: PartialOrd + PartialEq
{
    let high_ = high.unwrap_or_else(|| list_input.len() - 1);

    if low > high_{
        return low;
    }

    let mid = (low + high_) / 2;

    if value <= &list_input[mid]  {
        if mid == 0 {
            return 0;
        }

        bisect_rust_imp(list_input, value, low, Some(mid - 1))
    } else {
        bisect_rust_imp(list_input, value, mid+1, Some(high_))
    }
}


#[pyfunction]
pub fn bisect_rust(list_input: Vec<f64>, value: f64) -> usize {
    bisect_rust_imp(&list_input[..], &value, 0, None)
}

#[pyfunction]
pub fn bisect_rust_np(list_input: PyReadonlyArrayDyn<f64>, value: f64
) -> usize {
    let (x_vec, _o) = list_input.as_array().as_standard_layout().into_owned().into_raw_vec_and_offset();       
    bisect_rust_imp(&x_vec, &value, 0, None)
}    


#[pyfunction]
pub fn bisect_rust_arrow(array: PyArrowType<ArrayData>, value: f64) -> usize {
    let array = array.0; // Extract from PyArrowType wrapper
    let array: Arc<dyn Array> = make_array(array); // Convert ArrayData to ArrayRef
    let vw = array.as_primitive::<arrow::datatypes::Float64Type>().values();
    
    bisect_rust_imp(vw, &value, 0, None)
}


#[pyfunction]
pub fn bisect_rust_arrow_with_targets(source: PyArrowType<ArrayData>, targets: PyArrowType<ArrayData>) -> PyResult<PyArrowType<ArrayData>> {
    let source_array: Arc<dyn Array> = make_array(source.0); 
    let source_vw = source_array.as_primitive::<arrow::datatypes::Float64Type>().values();

    let target_array: Arc<dyn Array> = make_array(targets.0);
    let target_view = target_array.as_primitive::<arrow::datatypes::Float64Type>().values();

    let results_array: UInt64Array = target_view.iter().map(|x| bisect_rust_imp(source_vw, x, 0, None) as u64 ).collect();

    Ok(PyArrowType(results_array.into_data()))
}


/// A Python module implemented in Rust.
#[pymodule]
fn pysect(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bisect_rust, m)?)?;    
    m.add_function(wrap_pyfunction!(bisect_rust_np, m)?)?;    
    m.add_function(wrap_pyfunction!(bisect_rust_arrow, m)?)?;    
    m.add_function(wrap_pyfunction!(bisect_rust_arrow_with_targets, m)?)?;    
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
     fn test_int() {
        let v = vec![1, 2, 3, 4, 6, 8, 10];
        let s = 7;
        let r = bisect_rust_imp(&v, &s, 0, None);
        assert_eq!(r, 5);
     }

     #[test]
     fn test_float() {
        let v = vec![1., 5.5, 11., 19.3];
        assert_eq!(bisect_rust(v.clone(), 0.), 0);
        assert_eq!(bisect_rust(v.clone(), 20.), 4);
        assert_eq!(bisect_rust(v.clone(), 10.), 2);
     }

}