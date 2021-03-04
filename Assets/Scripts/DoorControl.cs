using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DoorControl : MonoBehaviour
{
    Animator animator;

    // Start is called before the first frame update
    void Start()
    {
        animator = GetComponentInChildren<Animator>();    
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Q)) {
            animator.SetTrigger("isOpen");
        } else 
        {
  //          animator.ResetTrigger("isOpen");
        }
    }
    // private void OnTriggerEnter()
    // {
    //     Collider collider = GetComponent<Collider>();
    //     if (collider.tag == "Player") {
    //        animator.SetTrigger("isOpen");
    //     } else 
    //     {
    //         animator.ResetTrigger("isOpen");
    //     }
    //  }
}
