(ns mlalgorithms.deep-learning.activation-functions
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.error :refer :all]))

(defprotocol PActivation
  (grad [this x]))

(defpyrecord Sigmoid []
  clojure.lang.IFn
  (invoke [this x]
          (/ 1 (+ 1 (exp (- x)))))

  PActivation
  (grad [this x]
        (* (this x) (- 1 (this x)))))

(defpyrecord Softmax []
  clojure.lang.IFn
  (invoke [this x]
          (let [e-x (->> x
                         (m/column-like (m/max x
                                               :axis -1
                                               :keepdims true))
                         (- x)
                         exp)]
            (/ e-x (m/column-like (m/sum e-x
                                         :axis -1
                                         :keepdims true)
                                  e-x))))

  PActivation
  (grad [this x]
        (let [p (this x)]
          (* p (- 1 p)))))

(defpyrecord TanH []
  clojure.lang.IFn
  (invoke [this x]
          (- (/ 2 (+ 1 (exp (* (- 2) x)))) 1))

  PActivation
  (grad [this x]
        (- 1 (pow (this x) 2))))

(defpyrecord ReLU []
  clojure.lang.IFn
  (invoke [this x]
          (m/where (ge x 0) x 0))

  PActivation
  (grad [this x]
        (m/where (ge x 0) 1 0)))

(defpyrecord LeakyReLU [(alpha 0.2)]
  clojure.lang.IFn
  (invoke [this x]
          (m/where (ge x 0) x (* alpha x)))

  PActivation
  (grad [this x]
        (m/where (ge x 0) 1 alpha)))

(defpyrecord ELU [(alpha 0.1)]
  clojure.lang.IFn
  (invoke [this x]
          (m/where (ge x 0) x (* alpha (- (exp x) 1))))

  PActivation
  (grad [this x]
        (m/where (ge x 0) 1 (+ (this x) alpha))))

(defpyrecord SELU [(alpha 1.6732632423543772848170429916717)
                   (scale 1.0507009873554804934193349852946)]
  clojure.lang.IFn
  (invoke [this x]
          (* scale
             (m/where (ge x 0) x (* alpha (- (exp x) 1)))))

  PActivation
  (grad [this x]
        (* scale
           (m/where (ge x 0) 1 (* alpha (exp x))))))

(defpyrecord SoftPlus []
  clojure.lang.IFn
  (invoke [this x]
          (log (+ 1 (exp x))))

  PActivation
  (grad [this x]
        (/ 1 (+ 1 (exp (- x))))))
