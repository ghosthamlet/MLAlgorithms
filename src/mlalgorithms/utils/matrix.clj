(ns mlalgorithms.utils.matrix
  (:refer-clojure :exclude [* - + / == <= >= not= min max empty])
  (:require [clojure.core.matrix :refer :all :as m :exclude [reshape]]
            [clojure.core.matrix.random :as random]
            [clojure.core.matrix.operators :refer :all :as op :exclude [max]]
            [clojure.core.matrix.linear :as l]
            [clojure.core.matrix.selection :as sel]
            [clojure.core.matrix.stats :as ms]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.error :refer :all])
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.ndarray INDArray]))

(defmacro with-indarray [[binding xs & kwargs] & body]
  `(let [~binding (tensor->indarray ~xs)
         kwargs# (apply array-map ~kwargs)
         ret# (matrix (do ~@body))]
     (if (contains kwargs# :axis)
       (do-keepdims ret#
                    :axis (:axis kwargs#)
                    :keepdims (:keepdims kwargs#))
       ret#)))

;; convert nd4j to vectorz is more diffcult
;; to default matrix just (matrix indarray)
;; so we have to use core.matrix default implment
;; https://www.programcreek.com/java-api-examples/index.php?api=org.nd4j.linalg.api.ndarray.INDArray
;; copy from https://github.com/yetanalytics/dl4clj/blob/master/src/nd4clj/linalg/factory/nd4j.clj
(defn vec->indarray
  [data]
  (Nd4j/create (double-array data)))

(defn matrix->indarray
  [matrix]
  (as-> (for [each matrix]
          (double-array each))
        data
    (into [] data)
    (into-array data)
    (Nd4j/create data)))

;; XXX: nd4j is more like numpy
;;      many fns can take axis/dims params
;;      and idx/axis/dims can be -1
(defn tensor->indarray
  [matrix]
  (if (isa? (class matrix) INDArray)
    matrix
    (let [dim (count (shape matrix))
          f3 #(let [m (pmap %2 %1)
                    a (Nd4j/zeros (int-array (shape %1)))]
                (dotimes [i (count m)]
                  (.putRow a i (nth m i)))
                a)]
      (case dim
        0 matrix
        1 (vec->indarray matrix)
        2 (matrix->indarray matrix)
        3 (f3 matrix matrix->indarray)
        4 (f3 matrix tensor->indarray)
        (not-implement)))))

;; support neg index
(defn mnth [xs n]
  (sel/sel xs
           (#(if (neg-int? %)
               (+ (count (vec xs)) %)
               %) n)
           (sel/irange)))

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

(defpy zeros [shape (dtype)]
  (case dtype
    :float (matrix (reshape [] shape))
    (reshape [] shape)))

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
(defpy sum [xs (axis) (keepdims)]
  (if (< 2 (count (shape xs)))
    (not-implement)
    (case axis
      0 (if keepdims
          [(apply map (fn [& xx] (apply + xx)) xs)]
          ;; (apply map (fn [& xx] (apply + xx)) xs)
          (esum xs))
      1 (if keepdims
          (map (fn [xx] [(apply + xx)]) xs)
          (map (fn [xx] (apply + xx)) xs))
      -1 (if keepdims
           (map (fn [xx] [(apply + xx)]) xs)
           (map (fn [xx] (apply + xx)) xs))
      (not-implement))))

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

(defpy insert-dim [xs (axis 0) (num 1)]
  (assert (> num 0) "dim num have to be > 0")
  (if (and (zero? axis) (= 1 num))
    [xs]
    (case axis
      0 (insert-dim [xs] axis (dec num))
      1 (pmap (fn [x] [x]) xs)
      2 (pmap #(insert-dim % :axis 1 :num num) xs)
      3 (pmap #(insert-dim % :axis 2 :num num) xs))))

(defpy do-keepdims [xs (axis 0) (keepdims true)]
  (if keepdims
    (insert-dim xs :axis axis :num 1)
    xs))

;; https://github.com/yetanalytics/dl4clj/blob/master/src/nd4clj/linalg/api/ndarray/indarray.clj
(defpy max [xs (axis 0) (keepdims false)]
  (with-indarray [m xs
                  :axis axis :keepdims keepdims]
    (.max m (int-array [axis]))))

(defpy amax [xs]
  (max xs))

(defpy where
  [condition x y]
  (eif condition x y)
  #_(if (= 2 (count (shape condition)))
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
  (case axis
    0 (apply map (fn [& a]
                   (index-of a (emax a)))
             xs)
    1 (map (fn [a]
             (index-of a (emax a)))
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
      1 (map concat xs1 xs2)
      (not-implement))
    (not-implement)))

(defpy normal [loc scale size]
  (reshape (repeatedly (apply * size)
                       #(random/rand-gaussian loc scale))
           size))

;; m/broadcast not like numpy
(defpy row-like [xs xs1]
  (repeat ((shape xs1) 0) (xs 0)))

(defpy column-like [xs xs1]
  (let [col-cnt (last (shape xs1))]
    (apply concat
           (emap #(repeat col-cnt %) xs))))

(defpy mean [xs (axis)]
  (case axis
    nil (ms/mean (flatten xs))
    (not-implement)))

;; like m[:, -1] in numpy
;; sel/end ?
(defpy -x [xs axis]
  (dec ((shape xs) axis)))

(defpy xi [xs i axis]
  (if (neg? i)
    (-x xs axis)
    i))

(defpy linspace [start stop
                 (num 50) (endpoint true)
                 (dtype)]
  (let [-div (if endpoint (- num 1) num)
        y (range num)
        delta (- stop start)
        step (/ delta -div)
        y (+ start
             (if (zero? step)
               (* (/ y -div) delta)
               (* y step)))
        y (if (and endpoint (> num 1))
            (assoc y (dec (count y)) stop)
            y)]
    (case dtype
      :float (matrix y)
      :int (map int y)
      (map int y))))

(defpy roll [xs shift (axis 0)]
  (roate xs axis shift)
  #_(if (= 2 (count (shape xs)))
      (case axis
        0 (let [absmove (mod shift (count (vec xs)))]
            (map-indexed (fn [i _]
                           (mnth xs (- i absmove)))
                         xs)
            #_(last (reduce (fn [[i xx] x]
                              [(inc i)
                               (assoc xx
                                      (mod (+ i shift) len) x)])
                            [0 xs]
                            xs)))
        (not-implement))
      (not-implement)))
