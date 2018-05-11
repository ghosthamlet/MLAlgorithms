(ns mlalgorithms.deep-learning.loss-functions
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]
            [mlalgorithms.protocols :as p]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.util :refer :all]
            [mlalgorithms.utils.error :refer :all]))

(defpyrecord SquareLoss []
  p/PLoss
  (loss [this y y-pred]
        (alog "loss")
        (alog "y: " (shape y) " y-pred: " (shape y-pred))
        (* 0.5 (pow (- y y-pred) 2)))

  (loss-grad [this y y-pred]
             (alog "loss-grad")
             (alog "y: " (shape y) " y-pred: " (shape y-pred))
             (- (- y y-pred))))

(defpyrecord CrossEntropy []
  p/PLoss
  (loss [this y y-pred]
        (alog "loss")
        (alog "y: " (shape y) " y-pred: " (shape y-pred))
        (let [y-pred (m/clip y-pred 1e-15 (- 1 1e-15))]
          (- (* (- y) (log y-pred))
             (* (- 1 y) (log (- 1 y-pred))))))

  (loss-grad [this y y-pred]
             (alog "loss-grad")
             (alog "y: " (shape y) " y-pred: " (shape y-pred))
             (let [y-pred (m/clip y-pred 1e-15 (- 1 1e-15))]
               (+ (- (/ y y-pred))
                  (/ (- 1 y) (- 1 y-pred)))))

  (acc [this y y-pred]
       (alog "acc")
       (alog "y: " (shape y) "y-pred: " (shape y-pred))
       (accuracy-score (m/argmax y :axis 1)
                       (m/argmax y-pred :axis 1))))
