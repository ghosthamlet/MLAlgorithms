(ns mlalgorithms.utils.error)

(defmacro not-implement []
  `(throw (Exception. "Not implement")))
