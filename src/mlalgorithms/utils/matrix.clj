(ns mlalgorithms.utils.matrix
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max empty])
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.random :as r]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :as l]
            [mlalgorithms.utils.error :refer :all]))

(defn uniform [low high size]
   (+ low
      (* (- high low)
         (r/sample-uniform size))))

 (defn insert [xs obj values & {:keys [axis]
                                :or {axis 1}}]
   (case axis
     1 (map #(let [[before after] (split-at obj %)]
               (concat before [values] after))
            xs)
     (not-implement)))

 (defn atleast-1d [xs]
   (if (shape xs) xs [xs]))

 (defn norm
   ([xs] (l/norm xs))
   ([xs order] (l/norm xs order))
   ([xs order axis]
    (let [x-shape-count (count (shape xs))]
      (if (= 1 x-shape-count)
        (l/norm xs order)
        (case axis
          -1 (case x-shape-count
               2 (map #(l/norm % order) xs)
               (not-implement))
          0 (case x-shape-count
              2 (apply map
                       (fn [& xs]
                         (l/norm xs order))
                       xs)
              (not-implement))
          nil (l/norm xs order)
          (not-implement))))))

 (defn expand-dims [xs axis]
   (case (count (shape xs))
     1 (case axis
         -1 (map (fn [x] [x]) xs)
         0 [xs]
         (not-implement))
     (not-implement)))

 (defn axis2-select
   "numpy: xs[:, idxes]
  deprecated! use instead:
  clojure.core.matrix.selection
  (sel/sel xs (sel/irange) idxes)
  (sel/set-sel xs (sel/irange) 1 [5 6])"
   [xs idxes]
   (map (fn [x]
          (map #(x %) idxes))
        xs))

 (defn prod [xs axis]
   (case (count (shape xs))
     2 (case axis
         -1 (map #(apply * %) xs)
         1 (map #(apply * %) xs)
         ;; use matrix not scalar calc
         ;; 0 (apply map * xs)
         0 (apply * xs)
         (not-implement))
     (not-implement)))

 (defn empty [shape]
   (reshape [] shape))
