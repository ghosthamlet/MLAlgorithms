(ns mlalgorithms.unsupervised-learning.generative-adversarial-network
  (:refer-clojure :exclude [* - + / == < <= > >= not= min max])
  (:require clojure.core.matrix.impl.ndarray
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.operators :refer :all]
            [clojure.core.matrix.random :as random]
            [clojure.core.matrix.selection :as sel]
            [mlalgorithms.protocols :as p]
            [mlalgorithms.deep-learning.neural-network :as nn]
            [mlalgorithms.deep-learning.layers :as layer]
            [mlalgorithms.deep-learning.optimizers :as optimizer]
            [mlalgorithms.deep-learning.loss-functions :as loss]
            [mlalgorithms.utils.code :refer :all]
            [mlalgorithms.utils.util :refer :all]
            [mlalgorithms.utils.matrix :as m]
            [mlalgorithms.utils.error :refer :all]))

(defpyrecord GAN
  [(img-rows 28) (img-cols 28)
   (img-dim) (latent-dim 100)
   (discriminator) (generator)
   (combined)]
  p/PGAN
  (init-gan-vars [this]
                 (assoc this
                        :img-dim (* img-rows img-cols)))

  (init-gan [this]
            (let [optimizer (optimizer/make-adam :learning-rate 0.0002
                                                 :b1 0.5)
                  loss-function (loss/make-crossentropy)
                  discriminator (p/build-discriminator this
                                                       optimizer
                                                       loss-function)
                  generator (p/build-generator this
                                               optimizer
                                               loss-function)]
              (alog)
              (nn/summary generator "Generator")
              (alog)
              (nn/summary discriminator "Discriminator")
              (assoc this
                     :discriminator discriminator
                     :generator generator
                     :combined (assoc (nn/make-neuralnetwork optimizer
                                                             :loss-function loss-function)
                                      :layers (concat (:layers generator)
                                                      (:layers discriminator))))))

  (build-generator [this optimizer loss-function]
                   (-> (nn/make-neuralnetwork optimizer :loss-function loss-function)
                       (nn/add-layer (layer/make-dense 256 :input-shape [latent-dim]))
                       (nn/add-layer (layer/make-activation :leaky-relu))
                       (nn/add-layer (layer/make-batchnormalization :momentum 0.8))
                       (nn/add-layer (layer/make-dense 512))
                       (nn/add-layer (layer/make-activation :leaky-relu))
                       (nn/add-layer (layer/make-batchnormalization :momentum 0.8))
                       (nn/add-layer (layer/make-dense 1024))
                       (nn/add-layer (layer/make-activation :leaky-relu))
                       (nn/add-layer (layer/make-batchnormalization :momentum 0.8))
                       (nn/add-layer (layer/make-dense img-dim))
                       (nn/add-layer (layer/make-activation :tanh))))

  (build-discriminator [this optimizer loss-function]
                       (-> (nn/make-neuralnetwork optimizer :loss-function loss-function)
                           (nn/add-layer (layer/make-dense 512 :input-shape [img-dim]))
                           (nn/add-layer (layer/make-activation :leaky-relu))
                           (nn/add-layer (layer/make-dropout :p 0.5))
                           (nn/add-layer (layer/make-dense 256))
                           (nn/add-layer (layer/make-activation :leaky-relu))
                           (nn/add-layer (layer/make-dropout :p 0.5))
                           (nn/add-layer (layer/make-dense 2))
                           (nn/add-layer (layer/make-activation :softmax))))

  (train [this n-epochs batch-size save-interval samples]
         ;; can use upto 15000
         (let [mnist (fetch-mldata :samples samples)
               ;; Rescale [-1, 1]
               _ (alog "Rescale")
               _ (alog (sel/sel (:data mnist) 0 0))
               X (/ (- (m/astype (:data mnist) :type :float32) 127.5) 127.5)
               y (:target mnist)
               half-batch (int (/ batch-size 2))]
           (alog "start")
           (loop [[epoch & epochs] (range n-epochs)]
             (if (nil? epoch)
               (alog "Train fin")
               ;; ---------------------
               ;;  Train Discriminator
               ;; ---------------------
               (let [_ (alog "Train Discriminator")
                     discriminator (nn/set-trainable discriminator true)
                     ;; Select a random half batch of images
                     idx (random/sample-rand-int half-batch ((shape X) 0))
                     imgs (sel/sel X idx (sel/irange))
                     ;; Sample noise to use as generator input
                     noise (m/normal 0 1 [half-batch latent-dim])
                     ;; Generate a half batch of images
                     gen-imgs (p/predict generator noise)
                     ;; Valid = [1, 0], Fake = [0, 1]
                     valid (m/concatenate (m/ones [half-batch 1])
                                          (m/zeros [half-batch 1])
                                          :axis 1)
                     fake (m/concatenate (m/zeros [half-batch 1])
                                         (m/ones [half-batch 1])
                                         :axis 1)
                     ;; Train the discriminator
                     [discriminator d-loss-real d-acc-real] (nn/train-on-batch discriminator imgs valid)
                     [discriminator d-loss-fake d-acc-fake] (nn/train-on-batch discriminator gen-imgs fake)
                     d-loss (* 0.5 (+ d-loss-real d-loss-fake))
                     d-acc (* 0.5 (+ d-acc-real d-acc-fake))
                     _ (alog)
                     _ (alog "Train Generator")
                     ;; ---------------------
                     ;;  Train Generator
                     ;; ---------------------
                     ;; We only want to train the generator for the combined model
                     discriminator (nn/set-trainable discriminator false)
                     ;; Sample noise and use as generator input
                     noise (m/normal 0 1 [batch-size latent-dim])
                     ;; The generator wants the discriminator to label the generated samples as valid
                     valid (m/concatenate (m/ones [batch-size 1])
                                          (m/zeros [batch-size 1])
                                          :axis 1)
                     ;; Train the generator
                     [combined g-loss g-acc] (nn/train-on-batch combined noise valid)]
                 (println (apply format "%d [D loss: %f, acc: %f] [G loss: %f, acc: %f]"
                                 epoch (map #(double %) [d-loss (* 100 d-acc) g-loss (* 100 g-acc)])))
                 ;; If at save interval => save generated image samples
                 #_(when (zero? (mod epoch save-interval))
                   (p/save-imgs this epoch))
                 (recur epochs))))))

  (save-imgs [this epoch]
             (let [[r c] [5 5] ; Grid size
                   noise (m/normal 0 1 [(* r c) latent-dim])
                   ;; Generate images and reshape to image shape
                   gen-imgs (m/reshape (p/predict generator noise)
                                       [-1 img-rows img-cols])
                   ;; Rescale images 0 - 1
                   gen-imgs (+ (* 0.5 gen-imgs) 0.5)]
               (spit (format "mnist_%d.png" epoch) gen-imgs))))

(defpy run [epochs (samples 10)]
  ;; (require '[mlalgorithms.unsupervised-learning.generative-adversarial-network :as gan] :reload-all)
  ;; (require '[mlalgorithms.protocols :as p])
  (time (p/train (-> (make-gan) p/init-gan-vars p/init-gan) epochs 64 400 samples)))
