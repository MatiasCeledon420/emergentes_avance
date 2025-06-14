using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MenuManager : MonoBehaviour {
    public GameObject back;
    public Transform init;

    private Stack<Transform> history = new Stack<Transform>();

    public void Awake() {
        ActivateMenu(init);
        back.SetActive(false);
    } 

    public Transform GetActiveMenu() {
        foreach (Transform iter in transform) {
            if ( iter.gameObject.activeSelf ) {
                return iter;
            }
        }
        return null;
    }

    public void GoBack() {
        if ( history.Count > 0 ) {
            ActivateMenu(history.Pop());

            back.SetActive(( history.Count > 0 ));
        }
    }

    public void ChangeMenu(Transform child) {
        history.Push(GetActiveMenu());
        ActivateMenu(child);

        back.SetActive(true);
    }

    private void ActivateMenu(Transform child) {
        child.gameObject.SetActive(true);

        foreach (Transform iter in transform) {
            if ( iter != child.transform ) {
                iter.gameObject.SetActive(false);
            }
        }
    }
}