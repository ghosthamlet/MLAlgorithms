(ns mlalgorithms.utils.code
  (:require [clojure.string :as s]))

(defmacro defpy
  "
  (defpy test [a (b 10) (c 20)]
    (+ a b c))

  (test 1 :b 2 :c 3)
  "
  [name all-args & body]
  (let [require-args (filter #(symbol? %) all-args)
        kwargs (remove #(symbol? %) all-args)
        args (reduce (fn [acc kw]
                       (assoc acc
                              :keys
                              (conj (:keys acc)
                                    (first kw))
                              :or
                              (assoc (:or acc)
                                     (first kw)
                                     (second kw))))
                     {:keys []
                      :or {}}
                     kwargs)]
    `(defn ~name [~@require-args & ~args]
       ~@body)))

#_(defmacro defpy
    "(defpy test [a (b 10) c] (+ a b c))
  (test :a 1 :b 2 :c 3)"
    [name kwargs body]
    (let [args (reduce (fn [acc kw]
                         (assoc acc
                                :keys
                                (conj (:keys acc)
                                      (if (symbol? kw)
                                        kw
                                        (first kw)))
                                :or
                                (if (symbol? kw)
                                  (:or acc)
                                  (assoc (:or acc)
                                         (first kw)
                                         (second kw)))))
                       {:keys []
                        :or {}
                        :as 'kwargs}
                       kwargs)
          require-args (filter #(symbol? %) kwargs)]
      `(defn ~name [& ~args]
         (doseq [a# '~require-args]
           (when-not (some #{(keyword a#)} (keys ~'kwargs))
             (throw (Exception. (str "Param :" a# " required")))))
         ~body)))

(defmacro defpyrecord
  "
  (defpyrecord RNN [alpha (lr 0) (w)]
      Layer
      (forword [])

      Model
      (fit []))

  (make-rnn 1 :lr 10 :w 0)
  "
  [name fields & specs]
  (let [keys (vec (map #(if (symbol? %)
                          %
                          (first %))
                       fields))]
    `(do
       (defrecord ~name ~keys
         ~@specs)

       (defpy ~(symbol (str "make-" (s/lower-case name)))
         ~fields
         (~(symbol (str name ".")) ~@keys)))))
