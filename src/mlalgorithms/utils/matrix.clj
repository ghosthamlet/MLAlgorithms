(ns mlalgorithms.utils.matrix
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max empty])
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.random :as r]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :as l]
            [clojure.core.matrix.selection :as sel]
            [mlalgorithms.utils.error :refer :all]))

(defmacro sety [xs x v]
  `(sel/set-sel ~xs (sel/irange) ~x ~v))

(defmacro gety [xs x]
  `(sel/sel ~xs (sel/irange) ~x))

(defmacro updatexs-in [xs ks f & args]
  `(sel/set-sel ~xs
                ~@ks
                (f (sel/sel ~xs ~@ks) ~@args)))

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

(defn zeros [shape]
  (reshape [] shape))

(defn zeros-like [xs]
  (reshape [] (shape xs)))

(defn empty [shape]
  (reshape [] shape))

(defn tile [xs reps])

(defn pad [xs pad-width mode & args])

;; use macro not fn just for ~@indices
(defmacro add-at [xs indices vs]
  `(if (= Long (type ~indices))
    (updatexs-in ~xs ~indices + ~vs)
    (let [freqs# (frequencies ~indices)]
      ;; TODO: accumulated results for elements that are indexed more than once like np.add.at
      (updatexs-in ~xs ~@indices + ~vs))))

;; clojure.core.matrix.stats/sum enhanced
(defn sum [xs axis keepdims]
  (case axis
    0 (if keepdims
        [(apply map (fn [& xx] (apply + xx)) xs)]
        (apply map (fn [& xx] (apply + xx)) xs))
    1 (if keepdims
        (map (fn [& xx] [(apply + xx)]) xs)
        (map (fn [& xx] (apply + xx)) xs))
    (not-implement)))
