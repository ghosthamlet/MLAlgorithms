(ns mlalgorithms.deep-learning.loss-functions
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.util :refer :all]
            [mlalgorithms.utils.error :refer :all]))

(defprotocol PLoss
  (loss [this y y-pred])
  (grad [this y y-pred])
  (acc [this y y-pred]))

(defpyrecord SquareLoss []
  PLoss
  (loss [this y y-pred]
        (* 0.5 (pow (- y y-pred) 2)))

  (grad [this y y-pred]
        (- (- y y-pred))))

(defpyrecord CrossEntropy []
  PLoss
  (loss [this y y-pred]
        (let [y-pred (m/clip y-pred 1e-15 (- 1 1e-15))]
          (- (* (- y) (log y-pred))
             (* (- 1 y) (log (- 1 y-pred))))))

  (grad [this y y-pred]
        (let [y-pred (m/clip y-pred 1e-15 (- 1 1e-15))]
          (+ (- (/ y y-pred))
             (/ (- 1 y) (- 1 y-pred)))))

  (acc [this y y-pred]
       (accuracy-score (m/argmax y :axis 1)
                       (m/argmax y-pred :axis 1))))
