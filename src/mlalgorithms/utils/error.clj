(ns mlalgorithms.utils.error)

(defmacro not-implement [& msgs]
  `(throw (Exception. (str "Not implement: " ~@msgs))))
