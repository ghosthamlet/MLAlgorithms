(ns mlalgorithms.supervised-learning.regression
  "Port from
  https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/regression.py"
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix.stats :refer :all]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.random :as r]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :as linear]
            [mlalgorithms.utils.matrix :as mm]))

(defprotocol PRegularization
  (grad [this w]))

(defrecord ZeroRegularization []
  clojure.lang.IFn
  (invoke [_ _] 0)

  PRegularization
  (grad [_ _] 0))

;; PRegularization for Lasso Regression
(defrecord L1Regularization [alpha]
  clojure.lang.IFn
  (invoke [this w]
    (-> w linear/norm (* alpha)))

  PRegularization 
  (grad [this w]
    (* alpha (signum w))))

;; PRegularization for Ridge Regression
(defrecord L2Regularization [alpha]
  clojure.lang.IFn
  (invoke [this w]
    (-> w transpose (dot w) (* alpha 0.5)))

  PRegularization
  (grad [this w]
    (* alpha w)))

;; PRegularization for Elastic Net Regression
;; l1-ratio 0.5
(defrecord L1L2Regularization [alpha l1-ratio]
  clojure.lang.IFn
  (invoke [this w]
    (* alpha
       (+ (-> w linear/norm (* l1-ratio))
          (-> w transpose (dot w) (* (- 1 l1-ratio) 0.5)))))

  PRegularization
  (grad [this w]
    (* alpha (signum w))))

(defmulti init-weights!
  (fn [model _] (type model)))

;; uniform use random, it has side effect
;; Initialize weights randomly [-1/N, 1/N]
(defmethod init-weights! :default [model n-features]
  (let [limit (/ 1 (sqrt n-features))]
    (mm/uniform (- limit) limit [n-features])))

(defn reg-fit! [{:keys [n-iterations
                       learning-rate
                       regularization]
                :as model}
               X y]
  (let [X (mm/insert X 0 1 :axis 1)]
    ;; Do gradient descent for n_iterations
    (loop [i n-iterations
           w (init-weights! model ((shape X) 1))
           training-errors []]
      (if (= i 0)
        (assoc model
               :w w
               :training-errors training-errors)
        (let [y-pred (dot X w)]
          (recur (dec i)
                 (- w (* learning-rate
                         ;; Gradient of l2 loss w.r.t w
                         (+ (dot (- y y-pred) X)
                            (grad regularization w))))
                 (conj training-errors
                       ;; Calculate l2 loss
                       (mean (+ (* 0.5 (pow (- y y-pred) 2))
                                (regularization w))))))))))

(defn reg-predict [model X]
  ;; Insert constant ones for bias weights
  (dot (mm/insert X 0 1 :axis 1) (:w model)))

(defprotocol PModel
  (fit [this X y])
  (predict [this X]))

(defrecord LinearRegression
    [n-iterations learning-rate
     regularization
     w training-errors]
  PModel
  (fit [this X y]
    (reg-fit! this X y))

  (predict [this X]
    (reg-predict this X)))

(defn make-regression
  [& {:keys [n-iterations learning-rate
             regularization
             w training-errors]
      :or {w nil}}]
  (LinearRegression. n-iterations learning-rate
                    regularization
                    w training-errors))

(defn test []
  (def X [[1 2] [10 20]])
  (def y [2 3])
  ;; same as ZeroRegularization
  (def zero-regularization (reify clojure.lang.IFn
                             (invoke [_ _] 0)
                             PRegularization
                             (grad [_ _] 0)))
  (def reg (make-regression :n-iterations 10
                            :learning-rate 0.001
                            :regularization zero-regularization))
  (def reg1 (fit reg X y))
  (predict reg1 X)
  (:w reg1)
  ; (:training-errors reg1)
  )

(test)
