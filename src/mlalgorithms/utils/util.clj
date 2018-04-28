(ns mlalgorithms.utils.util
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require [clojure.string :as s]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.random :as r]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :as l]
            [clojure.core.matrix.selection :as sel]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]))

(defn combinations-with-replacement [coll k]
  (when-not (zero? k)
    (when-let [[x & xs] coll]
      (if (= k 1)
        (map list coll)
        (concat (map (partial cons x)
                     (combinations-with-replacement coll (dec k)))
                (combinations-with-replacement xs k))))))

(defn normalize
  ([xs] (normalize xs 2 -1))
  ([xs order axis]
   (let [l2 (m/atleast-1d (m/norm xs order axis))
         l2 (map #(if (= 0 %) 1 %) l2)]
     ;; numpy can auto expand [[1 2] [2 3]] / [[1] [2]]
     ;; to / [[1 1] [2 2]]
     (/ xs (map #(apply repeat ((shape xs) 1) %)
                (m/expand-dims l2 axis))))))

(defn polynomial-features [xs degree]
  (let [[n-samples n-features] (shape xs)
        index-combinations
        (fn []
          (reduce (fn [acc v]
                    (concat acc v))
                  []
                  (reduce #(conj %1
                                 (combinations-with-replacement
                                  (range n-features) %2))
                          []
                          (range (inc degree)))))
        combinations (index-combinations)
        n-output-features (count combinations)
        xs (map #(map double %) xs)]
    (loop [i 0
           combinations combinations
           xs-new (m/empty [n-samples n-output-features])]
      (if (empty? combinations)
        xs-new
        (recur (inc i)
               (next combinations)
               (sel/set-sel xs-new
                            (sel/irange)
                            i
                            (m/prod (sel/sel xs
                                             (sel/irange)
                                             (first combinations))
                                    1)))))))

(defmacro gen-record
  "
  (gen-record RNN [alpha (lr 0) w]
      Layer
      (forword [])

      Model
      (fit []))
  "
  [name fields & specs]
  (let [keys []
        as []]
    `(do
       (defrecord ~name ~keys
         ~@specs)

       (defn ~(symbol (str "make-" (s/lower-case name)))
         [& {:keys ~keys
             :as ~as}]
         (~(symbol (str name ".")) ~keys)))))
