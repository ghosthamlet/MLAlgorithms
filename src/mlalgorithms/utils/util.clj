(ns mlalgorithms.utils.util
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require [clojure.string :as s]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.random :as r]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :as l]
            [clojure.core.matrix.selection :as sel]
            [clojure.java.io :as io]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]))

;; (def data-site "http://mldata.org/repository/data/download/csv/")
(def data-url "https://pjreddie.com/media/files/mnist_train.csv")

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
               (m/sety xs-new
                       i
                       (m/prod (m/gety xs (first combinations))
                               1)))))))

(defn accuracy-score [y y-pred]
  (/ (m/sum (= y y-pred) :axis 0) (count y)))

(defpy fetch-mldata [(name "mnist_train") (samples 10)]
  (let [filename name
        filepath (str "/tmp/" filename ".csv")
        file (io/file filepath)
        urlname data-url
        _ (prn "samples: " samples)
        xs (take samples
                 (s/split (if (.exists file)
                            (slurp file)
                            (let [content (slurp urlname)]
                              (spit file content)
                              content))
                          #"\n"))
        _ (prn "samples end")
        xs (map (fn [x]
                  (map #(Float/parseFloat %)
                       (s/split x #","))) xs)]
    {:data (map #(drop 1 %) xs)
     :target (map #(first %) xs)}))

#_(defn fetch-mldata [name]
  (let [filename (s/replace (s/lower-case name) " " "-")
        filepath (str "/tmp/" filename ".csv")
        file (io/file filepath)
        urlname (str data-site filename)]
    (if (.exists file)
      (slurp file)
      (let [content (slurp urlname)]
        (spit file content)
        content))))
