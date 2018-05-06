(ns mlalgorithms.utils.matrix
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max empty])
  (:require [clojure.core.matrix :refer :all]
            [clojure.core.matrix.random :as r]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :as l]
            [clojure.core.matrix.selection :as sel]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.error :refer :all]))

(defmacro sety [xs x v]
  `(sel/set-sel ~xs (sel/irange) ~x ~v))

(defmacro gety [xs x]
  `(sel/sel ~xs (sel/irange) ~x))

(defmacro updatexs-in [xs ks f & args]
  `(if (= Long (type ~ks))
     (not-implement "updatexs-in by scalar")
     (sel/set-sel ~xs ~@ks (~f (sel/sel ~xs ~@ks) ~@args))))

(defn reshape [xs newshape]
  (if (> (count (filter #(= -1 %) newshape)) 1)
    (throw (Exception. "Newshape can only have one unknown as -1"))
    (shape xs
           (if (= Long (type newshape))
             newshape
             (map #(if (= -1 %)
                     (/ (apply * (shape xs))
                        (abs (apply * newshape)))
                     %)
                  newshape)))))

(defn uniform [low high size]
   (+ low
      (* (- high low)
         (r/sample-uniform size))))

(defpy insert [xs obj values (axis 1)]
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

(defn tile [xs reps]
  (if (and (int? reps)
           (= 1 (count (shape xs))))
    (apply concat (repeat xs reps))
    (not-implement)))

(defn pad [xs pad-width mode]
  (if (= "constant" mode)
    (if (= 4
           (count (shape xs))
           (count pad-width))
      (if (= [[0 0] [0 0]]
             (subvec pad-width 0 2))
        (let [pad-s2-start-cnt (get-in pad-width [2 0])
              pad-s2-end-cnt (get-in pad-width [2 1])
              pad-s3-start (repeat (get-in pad-width [3 0]) 0)
              pad-s3-end (repeat (get-in pad-width [3 1]) 0)
              pad-s2 (repeat (+ (count pad-s3-start)
                                (count pad-s3-end)
                                ((shape xs) 3)) 0)
              pad-s2-start (repeat pad-s2-start-cnt pad-s2)
              pad-s2-end (repeat pad-s2-end-cnt pad-s2)
              pad-s3-fn #(vec (concat pad-s3-start % pad-s3-end))
              pad-s2-fn #(vec (concat pad-s2-start
                                      (map pad-s3-fn %)
                                      pad-s2-end))]
          (map (fn [s1] (map pad-s2-fn s1)) xs))
        (not-implement))
      (not-implement))
    (not-implement)))

;; use macro not fn just for ~@indices
;; FIXME: numpy like a[[1, 2]] failed, just can [slice(None), 2]
(defmacro add-at [xs indices vs]
  `(if (= Long (type ~indices))
     ;; (updatexs-in ~xs ~indices + ~vs)
     (not-implement "add-at by scalar")
     (let [freqs# (frequencies ~indices)]
       ;; TODO: accumulated results for elements that are indexed more than once like np.add.at
       (updatexs-in ~xs ~indices + ~@vs))))

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

(defn np-repeat [xs times])
