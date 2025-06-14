using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneLoader : MonoBehaviour {
    public void LoadScene(string name) {
        SceneManager.LoadScene(name);
    }

    public void LoadScene(int index) {
        SceneManager.LoadScene(index);
    }
}