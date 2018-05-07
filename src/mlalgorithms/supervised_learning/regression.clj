(ns mlalgorithms.supervised-learning.regression
  "Port from
  https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/supervised_learning/regression.py"
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix.stats :refer :all]
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.random :as r]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.linear :as l]
            [mlalgorithms.protocols :refer :all]
            [mlalgorithms.utils.util :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]))

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
    (-> w l/norm (* alpha)))

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
       (+ (-> w l/norm (* l1-ratio))
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
    (m/uniform (- limit) limit [n-features])))

(defn reg-fit! [{:keys [n-iterations
                       learning-rate
                       regularization]
                :as model}
               X y]
  (let [X (m/insert X 0 1 :axis 1)]
    ;; Do gradient descent for n-iterations
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
  (dot (m/insert X 0 1 :axis 1) (:w model)))

(defrecord LinearRegression
    [n-iterations learning-rate regularization
     gradient-descent w training-errors]
  PModel
  (fit [this X y]
    (if gradient-descent
      (reg-fit! this X y)
      (not-implement)))

  (predict [this X]
    (reg-predict this X)))

(defn lasso-normalize [X degree]
  (normalize (polynomial-features X degree)))

(defrecord LassoRegression
    [n-iterations learning-rate regularization
     degree w training-errors]
  PModel
  (fit [this X y]
    (reg-fit! this (lasso-normalize X degree) y))

  (predict [this X]
    (reg-predict this (lasso-normalize X degree))))

(defrecord PolynomialRegression
    [n-iterations learning-rate regularization
     degree w training-errors]
  PModel
  (fit [this X y]
    (reg-fit! this (polynomial-features X degree) y))

  (predict [this X]
    (reg-predict this (polynomial-features X degree))))

(defrecord RidgeRegression
    [n-iterations learning-rate regularization
     degree w training-errors]
  PModel
  (fit [this X y]
    (reg-fit! this X y))

  (predict [this X]
    (reg-predict this X)))

(defrecord PolynomialRidgeRegression
    [n-iterations learning-rate regularization
     degree w training-errors]
  PModel
  (fit [this X y]
    (reg-fit! this (lasso-normalize X degree) y))

  (predict [this X]
    (reg-predict this (lasso-normalize X degree))))

(defrecord ElasticNet
    [n-iterations learning-rate regularization
     degree w training-errors]
  PModel
  (fit [this X y]
    (reg-fit! this (lasso-normalize X degree) y))

  (predict [this X]
    (reg-predict this (lasso-normalize X degree))))

(defn make-linear-regression
  [& {:keys [n-iterations learning-rate
             gradient-descent w training-errors]
      :or {n-iterations 100
           learning-rate 0.001
           gradient-descent true}}]
  (LinearRegression. n-iterations learning-rate
                     ;; same as ZeroRegularization
                     (reify clojure.lang.IFn
                       (invoke [_ _] 0)
                       PRegularization
                       (grad [_ _] 0)) gradient-descent
                     w training-errors))

(defn make-lasso-regression
  [& {:keys [n-iterations learning-rate
             degree reg-factor w training-errors]
      :or {n-iterations 300
           learning-rate 0.01}}]
  (LassoRegression. n-iterations learning-rate
                   (L1Regularization. reg-factor)
                   degree w training-errors))

(defn reg-test []
  (def X [[1 2] [10 20]])
  (def y [2 3])
  (def lr (make-linear-regression :n-iterations 10
                                   :learning-rate 0.001))
  (def lr2 (fit lr X y))
  (predict lr2 X)
  (:w lr2)
  #_(:training-errors lr2))
