(ns mlalgorithms.utils.matrix
  (:refer-clojure :exclude [* - + / == <= >= not= min max empty])
  (:require [clojure.core.matrix :refer :all :as m :exclude [reshape]]
            [clojure.core.matrix.random :as random]
            [clojure.core.matrix.operators :refer :all :as op :exclude [max]]
            [clojure.core.matrix.linear :as l]
            [clojure.core.matrix.selection :as sel]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.error :refer :all]))

(defmacro sety [xs x v]
  `(sel/set-sel ~xs (sel/irange) ~x ~v))

(defmacro gety [xs x]
  `(sel/sel ~xs (sel/irange) ~x))

(defmacro updatexs-in [xs ks f & args]
  `(if (number? ~ks)
     (not-implement "updatexs-in by scalar")
     (sel/set-sel ~xs ~@ks (~f (sel/sel ~xs ~@ks) ~@args))))

(defn reshape [xs newshape]
  (if (> (count (filter #(= -1 %) newshape)) 1)
    (throw (Exception. "Newshape can only have one unknown as -1"))
    (m/reshape xs
               (if (number? newshape)
                 newshape
                 (map #(if (= -1 %)
                         (int (floor (/ (apply * (shape xs))
                                        (abs (apply * newshape)))))
                         %)
                      newshape)))))

(defn uniform [low high size]
  (+ low
     (* (- high low)
        (random/sample-uniform size))))

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

(defpy prod [xs (axis 0)]
  (case (count (shape xs))
    ;; 2
    1 (case axis
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

(defn ones [shape]
  (reshape (repeat (apply * shape) 1)
           shape))

(defn tile [xs reps]
  (if (and (int? reps)
           (= 1 (count (shape xs))))
    (apply concat (repeat xs reps))
    (not-implement)))

(defpy pad [xs pad-width
            (mode "constant") (constant-values 0)]
  (if (= "constant" mode)
    (if (= 4
           (count (shape xs))
           (count pad-width))
      (if (= [[0 0] [0 0]]
             (subvec pad-width 0 2))
        (let [pad-s2-start-cnt (get-in pad-width [2 0])
              pad-s2-end-cnt (get-in pad-width [2 1])
              pad-s3-start (repeat (get-in pad-width [3 0])
                                   constant-values)
              pad-s3-end (repeat (get-in pad-width [3 1])
                                 constant-values)
              pad-s2 (repeat (+ (count pad-s3-start)
                                (count pad-s3-end)
                                ((shape xs) 3))
                             constant-values)
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
  `(if (number? ~indices)
     ;; (updatexs-in ~xs ~indices + ~vs)
     (not-implement "add-at by scalar")
     (let [freqs# (frequencies ~indices)]
       ;; TODO: accumulated results for elements that are indexed more than once like np.add.at
       (updatexs-in ~xs ~indices + ~vs))))

#_(defn add-at [xs indices vs]
    (if (= Long (type ~indices))
    ;; (updatexs-in ~xs ~indices + ~vs)
      (not-implement "add-at by scalar")
      (let [freqs# (frequencies ~indices)]
      ;; TODO: accumulated results for elements that are indexed more than once like np.add.at
      ;; How to nest unquote-splicing?
        `(~(updatexs-in xs ~@indices + ~@vs)))))

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

(defpy np-repeat [xs times (axis)]
  (case axis
    2 (map (fn [a]
             (map (fn [b]
                    (mapcat #(repeat times %) b))
                  a))
           xs)
    3 (map (fn [a]
             (map (fn [b]
                    (map (fn [c]
                           (mapcat #(repeat times %) c)
                           b))
                    a))
             xs)
           (flatten (map #(repeat times %)
                         (flatten xs))))))

(defpy max [xs (axis 0) (keepdims false)]
  (case axis
    0 (if keepdims
        [(apply op/max xs)]
        (apply op/max xs))
    -1 (if keepdims
         (map (fn [x] [(apply op/max x)]) xs)
         (map (fn [x] (apply op/max x)) xs))
    (not-implement)))

;; juse can do like:
;;   (where (ge x 0) x 0)
;;   (where (ge x 0) 0 1)
(defpy where
  [condition x y]
  (if (= 2 (count (shape condition)))
    (if (and (number? x) (number? y))
      (map (fn [a]
             (map #(if (= 1 %) x y) a)) condition)
      (map (fn [ax by]
             (map (fn [a b]
                    (if (= b 1)
                      (if (number? y) a x)
                      (if (number? y) y a)))
                  ax
                  by))
           (if (number? y) x y)
           condition))
    (not-implement)))

(defpy index-of [xs e]
  (first (keep-indexed #(if (= e %2) %1) xs)))

(defpy argmax [xs (axis)]
  (if (zero? axis)
    (apply map (fn [& a]
                 (index-of a (apply max a)))
           xs)
    (not-implement)))

(defpy size [xs]
  (apply * (shape xs)))

(defpy clip [xs min max]
  (let [f (fn [a]
            (map #(cond
                    (> % max) max
                    (< % min) min
                    :else %)
                 a))]
    (case (count (shape xs))
      1 (f xs)
      2 (map f xs)
      (not-implement))))

(defpy any [xs]
  (not (every? zero? (flatten xs))))

(defpy astype [xs (type)]
  xs)

(defpy concatenate [xs1 xs2 (axis)]
  (if (= (shape xs1)
         (shape xs2))
    (case axis
      1 (map #(concat %1 %2) xs1 xs2)
      (not-implement))
    (not-implement)))

(defpy normal [loc scale size]
  (reshape (repeatedly (apply * size)
                       #(random/rand-gaussian loc scale))
           size))

;; m/broadcast not like numpy
(defpy row-like [xs xs1]
  (repeat ((shape xs1) 0) (xs 0)))
