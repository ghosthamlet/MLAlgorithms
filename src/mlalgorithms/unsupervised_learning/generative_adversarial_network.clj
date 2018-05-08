(ns mlalgorithms.unsupervised-learning.generative-adversarial-network
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]
            [mlalgorithms.protocols :as p]
            [mlalgorithms.deep-learning.neural-network :as nn]
            [mlalgorithms.deep-learning.layers :as layer]
            [mlalgorithms.deep-learning.optimizers :as optimizer]
            [mlalgorithms.deep-learning.loss-functions :as loss]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]))

(defpyrecord GAN
  [(img-rows 28) (img-cols 28)
   (img-dim) (latent-dim 100)
   (discriminator) (generator)
   (combined)]
  p/PGAN
  (init [this]
        (let [optimizer (optimizer/make-adam :learning-rate 0.0002
                                             :b1 0.5)
              loss-function (loss/make-crossentropy)
              discriminator (build-discriminator this
                                                 optimizer
                                                 loss-function)
              generator (build-generator this
                                         optimizer
                                         loss-function)]
          (prn)
          (summary generator "Generator")
          (summary discriminator "Discriminator")
          (assoc this
                 :img-dim (* img-rows img-cols)
                 :discriminator discriminator
                 :generator generator
                 :combined (assoc (nn/make-neuralnetwork optimizer
                                                         :loss loss-function)
                                  :layers (concat (:layers generator)
                                                  (:layers discriminator))))))

  (build-generator [this optimizer loss-function]
                   (-> (nn/make-neuralnetwork optimizer loss-function)
                       (add (layer/make-dense 256 :input-shape [latent-dim]))
                       (add (layer/make-activation :leaky-relu))
                       (add (layer/make-batchnormalization :momentum 0.8))
                       (add (layer/make-dense 512))
                       (add (layer/make-activation :leaky-relu))
                       (add (layer/make-batchnormalization :momentum 0.8))
                       (add (layer/make-dense 1024))
                       (add (layer/make-activation :leaky-relu))
                       (add (layer/make-batchnormalization :momentum 0.8))
                       (add (layer/make-dense img-dim))
                       (add (layer/make-activation :tanh))))

  (build-discriminator [this optimizer loss-function]
                       (-> (nn/make-neuralnetwork optimizer loss-function)
                           (add (layer/make-dense 512 :input-shape [img-dim]))
                           (add (layer/make-activation :leaky-relu))
                           (add (layer/make-dropout :p 0.5))
                           (add (layer/make-dense 256))
                           (add (layer/make-activation :leaky-relu))
                           (add (layer/make-dropout :p 0.5))
                           (add (layer/make-dense 2))
                           (add (layer/make-activation :softmax))))

  (train [this n-epochs batch-size save-interval]
         (let [mnist (fetch-mldata "MNIST original")
               ;; Rescale [-1, 1]
               X (/ (- (m/astype (:data mnist) :float32) 127.5) 127.5)
               y (:target mnist)
               half-batch (int (/ batch-size 2))]
           (loop [[epoch & epochs] (range n-epochs)]
             (if (nil? epochs)
               (prn "Train fin")
               ;; ---------------------
               ;;  Train Discriminator
               ;; ---------------------
               (let [discriminator (set-trainable discriminator true)
                     ;; Select a random half batch of images
                     idx (m/randint 0 ((shape X) 0) half-batch)
                     imgs (X idx)
                     ;; Sample noise to use as generator input
                     noise (m/normal 0 1 [half-batch latent-dim])
                     ;; Generate a half batch of images
                     gen-imgs (predict generator noise)
                     ;; Valid = [1, 0], Fake = [0, 1]
                     valid (m/concatenate (m/ones [half-batch 1])
                                          (m/zeros [half-batch 1])
                                          :axis 1)
                     fake (m/concatenate (m/zeros [half-batch 1])
                                         (m/ones [half-batch 1])
                                         :axis 1)
                     ;; Train the discriminator
                     [d-loss-real d-acc-real] (train-on-batch discriminator imgs valid)
                     [d-loss-fake d-acc-fake] (train-on-batch gen-imgs fake)
                     d-loss (* 0.5 (+ d-loss-real d-loss-fake))
                     d-acc (* 0.5 (+ d-acc-real d-acc-fake))
                     ;; ---------------------
                     ;;  Train Generator
                     ;; ---------------------
                     ;; We only want to train the generator for the combined model
                     discriminator (set-trainable discriminator false)
                     ;; Sample noise and use as generator input
                     noise (m/normal 0 1 [batch-size latent-dim])
                     ;; The generator wants the discriminator to label the generated samples as valid
                     valid (m/concatenate (m/ones [batch-size 1])
                                          (m/zeros [batch-size 1])
                                          :axis 1)
                     ;; Train the generator
                     [g-loss g-acc] (train-on-batch combined noise valid)]
                 (prn (str epoch
                           " [D loss: " d-loss
                           ", acc: " (* 100 d-acc)
                           "] [G loss: " g-loss
                           ", acc: " (* 100 g-acc)
                           "]"))
                 ;; If at save interval => save generated image samples
                 (when (zero? (% epoch save-interval))
                   (save-imgs this epoch))
                 (recur epochs))))))

  (save-imgs [this epoch]
             (let [[r c] [5 5] ; Grid size
                   noise (m/normal 0 1 [(* r c) latent-dim])
                   ;; Generate images and reshape to image shape
                   gen-imgs (m/reshape (predict generator noise)
                                       [-1 img-rows img-cols])
                   ;; Rescale images 0 - 1
                   gen-imgs (+ (* 0.5 gen-imgs) 0.5)]
               (spit (format "mnist_%d.png" epoch) gen-imgs))))

;; (train (make-gan) :n-epochs 200000 :batch-size 64 :save-interval 400)
