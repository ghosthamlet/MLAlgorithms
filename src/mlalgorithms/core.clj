(ns mlalgorithms.core
  (:refer-clojure :exclude [* - + / == < <= > >= not= = min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]))

;; http://spacemacs.org/layers/+lang/clojure/README.html
;; https://github.com/mikera/core.matrix/blob/develop/src/test/clojure/clojure/core/matrix/demo/pagerank.clj
(defn foo
  "I don't do a whole lot."
  [x]
  (println x "Hello, World!"))

;; interger can be just 64bit, but float no limit
(dot [[9999990.0 2000000.0 999910000.0] [10.0 9999999920.0 30.0]]
     [[999999999910000000000999999999910000000000]
      [9999999999992999999999900099999999999929999999999000]
      [99999999999999999300009999999999999999930000]])
