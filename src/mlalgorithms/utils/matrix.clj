(ns mlalgorithms.utils.matrix
  (:refer-clojure :exclude [* - + / == < <= > >= not= = min max])
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.random :as r]
            [clojure.core.matrix.operators :refer :all]
            [mlalgorithms.utils.error :refer :all]))

(defn uniform [low high size]
  (+ low
     (* (- high low)
        (r/sample-uniform size))))

(defn insert [arr obj values & {:keys [axis]
                                :or {axis 1}}]
  (case axis
    1 (map #(let [[before after] (split-at obj %)]
              (concat before [values] after))
           arr)
    (throw (not-implemention))))
