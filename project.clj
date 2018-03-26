(defproject mlalgorithmsclj "0.0.1-alpha"
  :description ""
  :url "https://github.com/ghosthamlet/MLAlgorithmsClj"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.9.0-alpha17"]
                 [net.mikera/core.matrix "0.62.0"]]

  :profiles {:dev {:source-paths ["src" "test/cljc" "test/clj"]}
             :test {:source-paths ["src" "test/cljc" "test/clj"]}})
